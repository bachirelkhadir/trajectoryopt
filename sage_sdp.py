from functools import reduce
import numpy as np
import picos
from sage.all import *


class SageSDPProblem:
    def __init__(self, list_vars, equality_constraints, psd_matrices, 
                 objective, debug=False):

        if debug:
            print('SDP with {} variables, involving {} matrices of sizes:'\
                  .format(len(list_vars), len(psd_matrices)))
            print(list(map(lambda M: M.dimensions()[0], psd_matrices)))
        self.prob = picos.Problem()
        self.list_vars = list_vars
        self.sage_equality_constraints = equality_constraints
        self.sage_psd_matrices = psd_matrices
        self.sage_objective = objective

        self._sage_to_pic = {}
        self._pic_equality_constraints = []
        self._pic_psd_constraints = []
        self._pic_objective = []

        self.build_picos_vars()
        self.build_equality_constraints()
        self.build_psd_matrices()
        self.build_objective()

    def _get_pic_var_from_sage_var(self, sage_var):
        return self._sage_to_pic.get(str(sage_var), None)

    
    def _get_pic_expr_from_sage_expr(self, sage_expr):
        sage_expr = SR(sage_expr)
        op = sage_expr.operands()
        # if sage_expr contains an operation
        if len(op) > 0:
            return sage_expr.operator()(*map(self._get_pic_expr_from_sage_expr, op))
        # otherwise, `sage_expr` is either a variable or a constant (i.e., a flot)
        else:
            pic_var = self._get_pic_var_from_sage_var(sage_expr)
            return pic_var if pic_var else float(sage_expr)
        
    def build_picos_vars(self):
        for v in self.list_vars:
            if not(self._get_pic_var_from_sage_var(v)):
                self._sage_to_pic[str(v)] = self.prob.add_variable(str(v))
            
    def build_equality_constraints(self):
        pic_lhs = map(self._get_pic_expr_from_sage_expr, self.sage_equality_constraints)

        for eq in pic_lhs:
            constraint = self.prob.add_constraint(eq == float(0))
            self._pic_equality_constraints.append(constraint)
                     
    def build_psd_matrices(self):
        for A in self.sage_psd_matrices:
            pic_A = [[self._get_pic_expr_from_sage_expr(Aij)
                      for Aij in Ai]
                     for Ai in A]
            if len(pic_A) == 1:
                self._pic_psd_constraints.append(pic_A[0][0] >= float(0))
                self.prob.add_constraint(pic_A[0][0] >= float(0))
                continue
            # build a picos matrix from pic_A
            pic_A = [reduce(lambda a, b: a & b, Ai) for Ai in pic_A]
            pic_A = reduce(lambda a, b: a // b, pic_A)
            #print(pic_A)
            self._pic_psd_constraints.append(pic_A >> float(0))
            self.prob.add_constraint(pic_A >> float(0))
        
    def build_objective(self):
        self._pic_objective = self._get_pic_expr_from_sage_expr(self.sage_objective)
        self.prob.set_objective('max', self._pic_objective)
        
    def solve(self,**solver_params):
        #status = self.prob.solve(solver='cvxopt', mosek_params={'MSK_IPAR_SIM_MAX_ITERATIONS': 100})['status']
        status = self.prob.solve(**solver_params)['status']
        return status, {k: v.value for k,v in self._sage_to_pic.items()}
    

def symb_matrix(n, name='Q'):
    template_var_names = name+"_%d%d"
    order = lambda idx: (max(idx), min(idx))
    var_names = map(lambda idx: template_var_names % order(idx), 
                    cartesian_product([range(1, n+1),]*2))
    var_names = np.matrix(var_names).reshape(n, n)
    create_sym_entry = np.vectorize(lambda name: polygen(QQ, name))
    Q = create_sym_entry(var_names)
    Q = matrix(SR, Q)
    return Q
