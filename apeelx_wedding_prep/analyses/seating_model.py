import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum, writeParams, setParam, read


class SeatingModel:

    RELATIONSHIP_MTX = pd.read_csv('../data/raw/Wedding Guest Network Data - Connection Matrix.csv', index_col=0)
    CONSTRAINT_MTX = pd.read_csv('../data/raw/Wedding Guest Network Data - Seating Constraints.csv', index_col=0)
    GUEST_LIST_DF = pd.read_csv('../data/raw/Wedding Guest Network Data - Guest List.csv', index_col=0)
    
    RELATIONSHIP_MTX.index = ['_'.join(g.split(' ')) for g in RELATIONSHIP_MTX.index]
    RELATIONSHIP_MTX.columns = ['_'.join(g.split(' ')) for g in RELATIONSHIP_MTX.columns]
    CONSTRAINT_MTX.index = ['_'.join(g.split(' ')) for g in CONSTRAINT_MTX.index]
    CONSTRAINT_MTX.columns = ['_'.join(g.split(' ')) for g in CONSTRAINT_MTX.columns]
    CONSTRAINT_MTX.fillna(0, inplace=True)
    GUEST_LIST_DF.index = RELATIONSHIP_MTX.index.values.tolist()

    def __init__(self,
        df_solution=None, guests_per_table=8, age_difference_penalty=0.1, 
        must_sit_together_score=100, age_diff_zero=6, base_score=0):

        self.guest_list = self.GUEST_LIST_DF.index
        self.age_difference_penalty = age_difference_penalty
        self.guests_per_table = guests_per_table
        self.must_sit_together_score = must_sit_together_score
        self.age_diff_zero = age_diff_zero
        self.base_score = base_score

        self.tables = self.get_tables()
        if df_solution is None:
            self.df_solution = self._make_solution_df(self.guest_list, self.tables.keys())
        else:
            self.df_solution = df_solution
            assert sorted(self.df_solution['guest'].unique()) == sorted(self.guest_list), 'guest list and solution do not match'
            for table in self.df_solution['table'].unique():
                self.tables[table] = len(self.df_solution[(self.df_solution['table'] == table) & (self.df_solution['solution'] == 1)])

        self.age_difference_mtx = self._get_age_difference_mtx()
        self.objective_cost_mtx, self.not_seated_together_list = self._get_objective_and_constraint_values()

    def create_model(self):
        self.model = Model()
        self.y = self.model.addVars(self.guest_list, self.tables.keys(), vtype=GRB.BINARY, name="y")
        self._set_objective()
        self.table_constraints, self.not_seated_together_constraints, self.one_table_assignment_constraints = self._set_constraints()
        self._set_start_values()

    def write_model(self, filename, include_start=False):
        self.model.write(f'../models/{filename}.mps')
        if include_start:
            self.model.write(f'../models/{filename}.mst')

    def get_tables(self):
        n_tables = len(self.guest_list)//self.guests_per_table
        extra_seats = len(self.guest_list) % self.guests_per_table
        print(f'n_tables: {n_tables}, extra_seats: {extra_seats}')
        tables = {}
        for t in range(n_tables):
            guests = self.guests_per_table + 1 if t < extra_seats else self.guests_per_table
            tables[f't_{t+1}'] = guests

        if not extra_seats:
            tables['t_1'] += 1
        
        return tables

    def _get_objective_and_constraint_values(self):
        objective_cost_mtx = pd.DataFrame(0, columns=self.guest_list, index=self.guest_list)
        not_seated_together_list = []
        for i, g in enumerate(self.guest_list):
            for gp in self.guest_list[i+1:]:
                objective_cost_mtx.loc[g, gp] = self.age_difference_penalty * (self.age_difference_mtx.loc[g, gp] - self.age_diff_zero) \
                    if self.age_difference_mtx.loc[g, gp] >= self.age_diff_zero else 0
                objective_cost_mtx.loc[g, gp] -= self.RELATIONSHIP_MTX.loc[g, gp] - self.base_score
                if self.CONSTRAINT_MTX.loc[g, gp] == 1:
                    objective_cost_mtx.loc[g, gp] -= self.must_sit_together_score

                elif self.CONSTRAINT_MTX.loc[g, gp] == -1:
                    not_seated_together_list.append(
                        (SeatingModel.convert_to_underscored(g), SeatingModel.convert_to_underscored(gp)))

        return objective_cost_mtx, not_seated_together_list

    def _get_age_difference_mtx(self):
        age_difference_mtx = pd.DataFrame(0, columns=self.guest_list, index=self.guest_list)
        for g in self.guest_list:
            for gp in self.guest_list:
                age_difference_mtx.loc[g, gp] = np.abs(self.GUEST_LIST_DF.loc[g, 'age'] - self.GUEST_LIST_DF.loc[gp, 'age'])

        return age_difference_mtx

    def _set_objective(self):
        self.model.setObjective(
            quicksum(
                quicksum(
                    quicksum(
                        self.objective_cost_mtx.loc[g, gp] * self.y[(g, table)] * self.y[(gp, table)] \
                            for table in self.tables.keys()) \
                        for gp in self.guest_list[i+1:]) \
                    for i, g in tqdm(enumerate(self.guest_list), total=len(self.guest_list))
            )
        )
        self.model.update()

    def _set_constraints(self):
        table_constraints = self.model.addConstrs(
            quicksum(self.y[(g, table)] for g in self.guest_list) <= max_seats \
                for table, max_seats in self.tables.items())
        not_seated_together_constraints = self.model.addConstrs(
            self.y[g, table] + self.y[gp, table] <= 1 \
                for table in self.tables.keys() \
                    for g, gp in self.not_seated_together_list)
        one_table_assignment_constraints = self.model.addConstrs(
            quicksum(self.y[g, table] for table in self.tables.keys()) == 1 \
                for g in self.guest_list)

        self.model.update()
        return table_constraints, not_seated_together_constraints, one_table_assignment_constraints

    @staticmethod
    def _make_solution_df(guest_list, tables):
        guest_array = []
        table_array = []
        for guest in guest_list:
            for table in tables:
                guest_array.append(guest)
                table_array.append(table)

        return pd.DataFrame({'guest': guest_array, 'table': table_array, 'solution': [0] * len(guest_array), 'fixed': [False] * len(guest_array)})

    def _set_start_values(self):
        solution_df = self.df_solution.set_index(['guest', 'table'])
        for var in self.model.getVars():
            guest, table = var.VarName.split('[')[1].split(']')[0].split(',')
            var.Start = solution_df.loc[(guest, table), 'solution']
            if solution_df.loc[(guest, table), 'fixed']:
                bound = ('ub', 0.0)
                if solution_df.loc[(guest, table), 'solution'] == 1:
                    bound = ('lb', 1.0)

                var.setAttr(bound[0], bound[1])

        self.model.update()

    @staticmethod
    def convert_to_underscored(string):
        return string.replace(' ', '_')

    @staticmethod
    def convert_to_spaced(string):
        return string.replace('_', ' ')

    @classmethod
    def read_solution_file(cls, filename):
        df_solution = cls._read_solution(filename)
        return cls(df_solution)

    @staticmethod
    def _read_solution(filename):
        with open(filename, newline='\n') as csvfile:
            reader = csv.reader((line.replace('  ', ' ') for line in csvfile), delimiter=' ')
            next(reader)  # skip header
            sol = {}
            for var, value in reader:
                sol[var] = float(value)

        guests = []
        tables = []
        solutions = []
        for idx in sol.keys():
            guest, table = idx.split('[')[1].split(']')[0].split(',')
            guests.append(guest)
            tables.append(table)
            solutions.append(sol[idx])

        return pd.DataFrame({'guest': guests, 'table': tables, 'solution': solutions, 'fixed': [False] * len(guests)})

    @staticmethod
    def process_solution(filename):
        df_solution = SeatingModel._read_solution(filename)
        solution = {t: [] for t in sorted(df_solution['table'].unique())}
        for idx, row in df_solution.iterrows():
            if row['solution'] == 1:
                solution[row.table].append(SeatingModel.convert_to_spaced(row.guest))

        max_guests_per_table = max([len(guests) for guests in solution.values()])
        for table in solution.keys():
            if len(solution[table]) != max_guests_per_table:
                deficit = max_guests_per_table - len(solution[table])
                solution[table] += [''] * deficit

        seating_chart = pd.DataFrame(solution)
        seating_chart.index = [f'Guest {i+1}' for i in range(max_guests_per_table)]
        return seating_chart

    @staticmethod
    def make_df_solution_from_seating_chart(seating_chart):
        guest_idx = [idx for idx in seating_chart.index if idx.startswith('Guest')]
        guests = seating_chart.loc[guest_idx].values.flatten().tolist()
        guests = [g for g in guests if isinstance(g, str) and g != '']
        guests = [SeatingModel.convert_to_underscored(g) for g in guests]
        locked_guests = []
        for guest in guests:
            if '(l)' in guest.lower():
                locked_guests.append('_'.join(guest.split('_')[:-1]))

        guests = [g for g in guests if '_'.join(g.split('_')[:-1]) not in locked_guests]
        guests += locked_guests

        tables = seating_chart.columns.tolist()
        df_solution = SeatingModel._make_solution_df(guests, tables)
        df_solution.set_index(['guest', 'table'], inplace=True)
        df_solution['fixed'] = False
        for table in tables:
            guests_at_table = seating_chart.loc[guest_idx, table].values.tolist()
            guests_at_table = [g for g in guests_at_table if isinstance(g, str) and g != '']
            guests_at_table = [g.replace(' (L)', '') for g in guests_at_table]
            guests_at_table = [SeatingModel.convert_to_underscored(g) for g in guests_at_table]
            df_solution.loc[(guests_at_table, table), 'solution'] = 1
            if isinstance(seating_chart.loc['Locked', table], str) and seating_chart.loc['Locked', table].lower() == 'y':
                df_solution.loc[(guests_at_table, table), 'fixed'] = True
                df_solution.loc[(guests, table), 'fixed'] = True

        for guest in locked_guests:
            df_solution.loc[(SeatingModel.convert_to_underscored(guest), tables), 'fixed'] = True
        
        return df_solution.reset_index()