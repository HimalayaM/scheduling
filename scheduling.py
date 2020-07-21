#Python CP_Model
#https://google.github.io/or-tools/python/ortools/sat/python/cp_model.html

#Google_OR for CP-SAT solver
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model
import pandas as pd

import sys
seconds = 5
if len(sys.argv) > 1:
    seconds = int(sys.argv[1])

class residentsPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, rotations, num_residents, num_weeks, num_rotations, sols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._rotations = rotations
        self._num_residents = num_residents
        self._num_weeks = num_weeks
        self._num_rotations = num_rotations
        self._solutions = set(sols)
        self._solution_count = 0
        self.sol = []

    def on_solution_callback(self):
        sol = [ [''] * self._num_weeks for _ in range(self._num_residents)] 
        if self._solution_count in self._solutions:
            print('Solution %i' % self._solution_count)
            for w in range(self._num_weeks):
                for r in range(self._num_residents):
                    for s in range(self._num_rotations):
                        if self.Value(self._rotations[r, s, w]):
                            sol[r][w] = s
            p = pd.DataFrame(sol)
            print(p)
            print()
        self._solution_count += 1

    def solution_count(self):
        return self._solution_count

def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.
  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.
  Args:
    works: a list of variables to extract the span from.
    start: the start to the span.
    length: the length of the span.
  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence

def add_soft_sequence_constraint(model, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """Sequence constraint on true variables with soft and hard bounds.
  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    prefix: a base name for penalty literals.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': under_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': over_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])

    return cost_literals, cost_coefficients

def add_soft_sum_constraint(model, works, hard_min, soft_min, min_cost,
                            soft_max, hard_max, max_cost, prefix):
    """Sum constraint with soft and hard bounds.
  This constraint counts the variables assigned to true from works.
  If forbids sum < hard_min or > hard_max.
  Then it creates penalty terms if the sum is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a sum of at least
      hard_min.
    soft_min: any sequence should have a sum of at least soft_min, or a linear
      penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the sum is less than
      soft_min.
    soft_max: any sequence should have a sum of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a sum of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the sum is more than
      soft_max.
    prefix: a base name for penalty variables.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_variables = []
    cost_coefficients = []
    sum_var = model.NewIntVar(hard_min, hard_max, '')
    # This adds the hard constraints on the sum.
    model.Add(sum_var == sum(works))

    # Penalize sums below the soft_min target.
    if soft_min > hard_min and min_cost > 0:
        delta = model.NewIntVar(-len(works), len(works), '')
        model.Add(delta == soft_min - sum_var)
        # TODO(user): Compare efficiency with only excess >= soft_min - sum_var.
        excess = model.NewIntVar(0, 7, prefix + ': under_sum')
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(min_cost)

    # Penalize sums above the soft_max target.
    if soft_max < hard_max and max_cost > 0:
        delta = model.NewIntVar(-7, 7, '')
        model.Add(delta == sum_var - soft_max)
        excess = model.NewIntVar(0, 7, prefix + ': over_sum')
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(max_cost)

    return cost_variables, cost_coefficients

def main():
    # Create the Google CP-SAT solver
    m = cp_model.CpModel()

    #Objective function penalties
    obj_int_vars = []
    obj_int_coeffs = []
    obj_bool_vars = []
    obj_bool_coeffs = []

    #Create Data
    num_residents = 40
    num_rotations = 16
    num_weeks = 12

    all_residents = range(num_residents)
    all_rotations = range(num_rotations)
    all_weeks = range(num_weeks)
    min_rotations_per_resident = 1

    #Rotation Ind Dictionary
    rotation_ind = {}
    for i in range(6):
        rotation_ind[i] = "service"
    for i in range(6, num_rotations):
        rotation_ind[i] = "elective"
    
    #Define decision variables
    shift = {}
    for r in range(num_residents):
        for s in range(num_rotations):
            for w in range(num_weeks):
                shift[r,s,w] = m.NewBoolVar('shift_%i_%i_%i' % (r, s, w))

    #Intermittant Variable
    
    #Whether on rotation or on vacation
    on_rotation = {}
    for r in all_residents:
        for w in all_weeks:
            on_rotation[r,w] = m.NewBoolVar('on_service_%i_%i' % (r, w))
    for r in all_residents:
        for w in all_weeks:
            m.Add((on_rotation[r,w] == sum([shift[r,s,w] for s in all_rotations])))
 
    #CONSTRAINTS
    
    #Each service has 2 residents per week, each elective has one or less
    for w in all_weeks:
        for s in all_rotations:
            if (rotation_ind[s] == "service"): m.Add(sum(shift[r,s,w] for r in all_residents) == 2)
            #if (rotation_ind[s] == "elective"): m.Add(sum(shift[r,s,w] for r in all_residents) <= 6)

    #A resident cannot be on more than one rotation in a given week
    for r in all_residents:
        for w in all_weeks:
            m.Add(sum(shift[r,s,w] for s in all_rotations) <= 1)

    #All residents work a minimum number of rotations        
    for r in all_residents:
            num_rotations_worked = sum(shift[r,s,w] for w in all_weeks for s in all_rotations)
            m.Add(min_rotations_per_resident <= num_rotations_worked)

        
    # Week work constraints on continuous sequence :
        #     (rotation_type, hard_min, soft_min, min_penalty,
        #             soft_max, hard_max, max_penalty)
    wk_work_constraints = [
        # One or two consecutive days of rest, this is a hard constraint.
        ("service", 2, 2, 0, 4, 4, 0),
        ("elective", 2, 2, 0, 2, 2, 0)
    ]
    
    # Weekly sum constraints on shifts days:
    #     (hard_min, soft_min, min_penalty,
    #             soft_max, hard_max, max_penalty)
    weekly_sum_constraints = [
        # Constraints on rests per year.
        (10, 11, 10, 11, 12, 10)
    ]

    # Forcing residents to be on service 2-4 weeks in a row, residents on elective to be there 4 weeks in a row  
    for s in all_rotations:
        if (rotation_ind[s] == "service"): rotation_type, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = wk_work_constraints[0]
        if (rotation_ind[s] == "elective"): rotation_type, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = wk_work_constraints[1]
        for r in all_residents:
            works = [shift[r,s,w] for w in range(num_weeks)]
            variables, coeffs = add_soft_sequence_constraint(m, works, hard_min, soft_min, min_cost, soft_max, hard_max,
                max_cost, 'shift_constraint(resident %i, service %i)' % (r, s))
    
    # Forcing residents to have 1 to 2 weeks of vacation  
    hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = weekly_sum_constraints[0]
    for r in all_residents:
        works = [shift[r,s,w] for w in range(num_weeks) for s in range(num_rotations)]
        variables, coeffs = add_soft_sum_constraint(m, works, hard_min, soft_min, min_cost, soft_max, hard_max,
            max_cost, 'shift_constraint(resident %i, service %i)' % (r, s))


    #Somehow limit total weeks worked. 
    #Range the number of hard services worked
    #Somehow show diversity in the services each resident works in
    #do not allow three weeks of service
    
    
    '''            
    #Each resident gets 1 week off for vacation
    for r in all_residents:
        m.Add(sum([on_rotation[r,w] for w in all_weeks]) == (num_weeks - 1))

    # Forcing all residents to not be on service for 3 weeks
    for s in all_rotations:
        for r in all_residents:
            works = [shift[r,s,w] for w in range(num_weeks)]
            for start in range(len(works) - 3 + 1): 
                m.AddBoolOr(negated_bounded_span(works, start, 3))

    
    # Forcing residents to have at least 4 week off every three weeks
    hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = wk_work_constraints[1]
    for r in all_residents:
        works = [on_service[r,w] for w in range(num_weeks)]
        variables, coeffs = add_soft_sequence_constraint(m, works, hard_min, soft_min, min_cost, 
                                                         soft_max, hard_max, max_cost, 'shift_constraint(resident %i)' % (r))
        obj_bool_vars.extend(variables)
        obj_bool_coeffs.extend(coeffs)

    '''

    #Objective Function
    # m.Minimize(sum(obj_bool_vars[i] * obj_bool_coeffs[i] for i in range(len(obj_bool_vars)))
    #    + sum(obj_int_vars[i] * obj_int_coeffs[i] for i in range(len(obj_int_vars))))            # -- uncomment this line to get one solution with object minimize fn

    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    # Sets a time limit of 10 seconds.
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(m)
    print('Status = %s' % solver.StatusName(status))
    # Display the first five solutions.
    a_few_solutions = range(5)
    solution_printer = residentsPartialSolutionPrinter(shift, num_residents,
                                                    num_weeks, num_rotations,
                                                    a_few_solutions)
    solver.SearchForAllSolutions(m, solution_printer)


    # solver.SolveWithSolutionCallback(m, solution_printer)   # -- uncomment this line to get one solution with object minimize fn

    # #Call the solver and display the results        
    # solver = cp_model.CpSolver()
    # solver.parameters.linearization_level = 0
    # # Sets a time limit of 10 seconds.
    # solver.parameters.max_time_in_seconds = 300
    # status = solver.Solve(m)
    # print('Status = %s' % solver.StatusName(status))

    # #Print variable solution
    # for r in all_residents:
    #     for s in all_rotations:
    #         for w in all_weeks:
    #             if (solver.Value(shift[r,s,w]) == 1):
    #                 print('(r,s,w,sol);%i;%i;%i;%i' % (r,s,w,solver.Value(shift[r,s,w])))  


if __name__ == '__main__':
    main()
