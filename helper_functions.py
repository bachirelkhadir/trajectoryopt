import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sage.all import *
from sage.symbolic.expression_conversions import ExpressionTreeWalker
from tqdm.notebook import tqdm, trange
import sage_cvxpy


class SubstituteNumericalApprox(ExpressionTreeWalker):
    def __init__(self, **kwds):
        """
        A class that walks the tree and replaces numbers by numerical
        approximations with the given keywords to `numerical_approx`.
        EXAMPLES::
            sage: var('F_A,X_H,X_K,Z_B')
            sage: expr = 0.0870000000000000*F_A + X_H + X_K + 0.706825181105366*Z_B - 0.753724599483948
            sage: SubstituteNumericalApprox(digits=3)(expr)
            0.0870*F_A + X_H + X_K + 0.707*Z_B - 0.754
        """
        self.kwds = kwds

    def pyobject(self, ex, obj):
        if hasattr(obj, 'numerical_approx'):
            return obj.numerical_approx(**self.kwds)
        else:
            return obj


def round_polynomial_expression(g):
    approx = SubstituteNumericalApprox(digits=2)
    return approx(SR(g))


def isnotebook():
    """
    Returns true if the code is being run from a Jupyter notebook.
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter



def exponent_monomials_up_to_deg(num_variables, max_deg):
    """
    Returns the exponents of all monomials in `num_variables`
    variables up to degree `max_deg` in a list.
    """
    if max_deg < 0:
        return [[0,]*num_variables]
    return sum(map(list, (WeightedIntegerVectors(d, [1,] *
                                                 num_variables) for d in range(max_deg + 1))), [])


def monomials_up_to_deg(x, max_deg_x):
    """
    All monomials in x of degree <= max_deg_x.
    """

    return [
        prod((xi**alpha_i for xi, alpha_i in zip(x,alpha)))
        for alpha in exponent_monomials_up_to_deg(len(x), max_deg_x)
    ]


# Create variable measures

def is_var_in_list(x, L):
    return str(x) in map(str, L)


def measure_in_vars(vars, max_deg, name):
    """
    Truncated moments in `vars`
    """
    mon_vars = monomials_up_to_deg(vars, max_deg)
    y = {SR(mon): var(name + '_' + ''.join(str(mon.degree(xi))
                                                  for xi in vars))
            for mon in mon_vars}
    y[1] = 1
    Ly = tv_linear_form_of_measure(vars, y)
    
    return y, Ly



def tv_linear_form_of_measure(vars, y):
    """
    Given the vector of moments y(t) (given as a dict, y(t)_alpha = y[t^k, x^alpha]),
    return the function Ly_t s.t.  Ly_t(f) = <y(t), f> for any polynomial f(x) of deg < max moment degree in y.
    """

    def get_moment_mon(m):
        params = [p for p in m.variables() if not is_var_in_list(p, vars)]
        coeff = prod([p**m.degree(p) for p in params])
        monom = prod([v**m.degree(v) for v in m.variables() if is_var_in_list(v, vars)])
        return y[SR(monom)] * coeff

    def Ly_t(f):
        f = SR(f)
        f_vars = set(map(str, f.variables() + tuple(vars)))
        f_vars = tuple(f_vars)
        f = QQ[f_vars](f)
        return sum(f.monomial_coefficient(m) * get_moment_mon(m)
                   for m in f.monomials()
                   )
    return Ly_t


def degree_in(g, vars):
    params = [p for p in g.variables() if not is_var_in_list(p, vars)]
    return max(m.degree() - sum(m.degree(vi) for vi in params) for m in g.monomials())


def loc_moment_matrix(vars, L, g, max_deg):
    """
    Localized moment matrix of L at g.
    """

    deg_g = ceil(degree_in(g, vars)/2)
    deg_g = max(deg_g, 0)
    half_mons = monomials_up_to_deg(vars, max_deg//2-deg_g)
    
    return Matrix(SR, 
                  [[L(SR(g)*mi*mj) for mi in half_mons] for mj in half_mons])



##############################
# TV-SDP
#############################


def symb_matrix(n, name='Q'):
    """Construct an `n` x `n` symmetric symbolic matrix named `name`."""
    template_var_names = name+"_%d%d"
    order = lambda idx: (max(idx), min(idx))
    var_names = map(lambda idx: template_var_names % order(idx), 
                    cartesian_product([range(1, n+1),]*2))
    var_names = list(var_names)
    var_names = np.matrix(var_names).reshape(n, n)
    create_sym_entry = np.vectorize(lambda name: polygen(QQ, name))
    Q = create_sym_entry(var_names)
    Q = matrix(SR, Q)
    return Q


def make_matrix_sos_on_0_1(M, name='Q'):
    """
    Make time-varying matrix M(t) sos on [0, 1].
    """

    var('t')
    n = M.dimensions()[0]
    u = [var('u_{}'.format(i)) for i in range(n)]
    poly_vars = (t,) + tuple(u)
    quad_form = vector(u) * M * vector(u)
    big_ring = QQ[quad_form.variables()]

    quad_form = big_ring(quad_form)
    deg_t = quad_form.degree(big_ring(t))
    
    if deg_t % 2 == 0:
        mon_ut1 = [ui * t**k for ui in u for k in range(deg_t//2+1)]
        Q1 = symb_matrix(len(mon_ut1), name+'_1')
        quad1 = vector(mon_ut1) * Q1 * vector(mon_ut1)
        
        mon_ut2 = [ui * t**k for ui in u for k in range(deg_t//2)]
        Q2 = symb_matrix(len(mon_ut2), name+'_2')
        quad2 = t*(1-t)*vector(mon_ut2) * Q2 * vector(mon_ut2)
        
    else:
        mon_ut = [ui * t**k for ui in u for k in range((deg_t-1)//2 + 1)]

        Q1 = symb_matrix(len(mon_ut), name+'_1')
        quad1 = t * vector(mon_ut) * Q1 * vector(mon_ut)
        
        Q2 = symb_matrix(len(mon_ut),  name+'_2')
        quad2 = (1-t)*vector(mon_ut) * Q2 * vector(mon_ut)
        
    residual = quad_form - quad1 - quad2
    poly_vars = (t,) + tuple(u)
    poly_vars = tuple(list(map(SR, poly_vars)))
    
    decision_vars = tuple(set(big_ring.gens()))

    decision_vars += Q1.variables()
    decision_vars += Q2.variables()
    decision_vars = tuple(list(map(SR, decision_vars)))
    decision_vars = set(decision_vars) - set(poly_vars)
    decision_vars = tuple(decision_vars)
        
    residual = QQ[decision_vars][poly_vars](residual)
    return {'linear_eq': residual.coefficients(), 'psd_matrices': [Q1, Q2]}




def plot_a_b(a, b):
    """
    Plot starting and ending points a and b.
    """

    plt.scatter(*zip(a,b), marker='x', color='k')    
    plt.annotate(r'$a$', list(np.array(a) + [0, -.1]), size=15)
    plt.annotate(r'$b$', list(np.array(b) + [0, .1]), size=15)
    
    

def plot_obstacle_and_trajectory(pieces_coefficients, obstacle_plot, a, b):
    fig = plt.figure()
    fig = obstacle_plot.matplotlib(figure=fig)
    for j, piece_coefficients in enumerate(pieces_coefficients):
        plt.plot(*extract_expected_trajectory(piece_coefficients, 0, 1).T, lw=3,  label=f'{j}', ls='--')
    plot_a_b(a, b)
    plt.legend()



def extract_expected_trajectory(piece_coefficients, t0=0, t1=1, num_samples=100):
    u1,u2,v1,v2 = piece_coefficients
    xt = lambda t: np.array([u1+t*u2, v1+t*v2])
    timestamps = np.linspace(t0, t1, num_samples)
    xt_sampled = np.array([xt(si) for si in timestamps])
    return xt_sampled



def prepare_sdp_problem(P, M, a, b, obstacle_equations, max_deg, num_pieces, ):
    t, x1, x2, u1, u2, v1, v2 = P.gens()
    

    # localization of the coefficients (u, v)
    linear_eq = []
    psd_matrices = []
    loc_mom_matrices = {}
    poly_vars = [u1,u2,v1,v2]
    x1 = u1+t*u2
    x2 = v1+t*v2


    phi_pieces = [measure_in_vars(poly_vars, max_deg, f'phi_{i}') 
                  for i in range(num_pieces)]
    phis = [phi for phi, _ in  phi_pieces]
    Lphis = [Lphi for _, Lphi in phi_pieces]


    # constraints


    # localization constraints
    loc_polynomials = [
       P(1),
        u1+M,     M-u1,      M**2-u1**2,        #M**4-u1**4,
        u1+u2+M,  M-(u1+u2), M**2-(u1+u2)**2,   #M**4-(u1+u2)**4,
        v1+M,     M-v1,      M**2-v1**2,        #M**4-v1**4,
        v1+v2+M,  M-(v1+v2), M**2-(v1+v2)**2,   #M**4-(v1+v2)**4,
       *[expand(g.subs(x1=x1, x2=x2)) for g in obstacle_equations]
    ]

    for i, Lphi in tqdm(enumerate(Lphis), desc='making constraints for piece', total=int(num_pieces)):
        for g in loc_polynomials:
            tag_name = f"loc_{i}({round_polynomial_expression(g)})"
            loc_mom_matrices[tag_name] =\
                loc_moment_matrix(poly_vars, Lphi, g, max_deg)




    # y(0) ~ a, y2(1) ~ b
    linear_eq +=sum([
        [Lphis[0](m * (x1.subs(t=0) - a[0]) ), Lphis[0](m * (x2.subs(t=0) - a[1]) )] 
         for m in monomials_up_to_deg(poly_vars, max_deg-1)], [])

    linear_eq +=sum([
        [Lphis[-1](m * (x1.subs(t=1) - b[0]) ), Lphis[-1](m * (x2.subs(t=1) - b[1]) )] 
         for m in monomials_up_to_deg(poly_vars, max_deg-1)], [])


    # continuity condition between pieces
    for L_t, Lplus_t in zip(Lphis[:-1], Lphis[1:]):
        linear_eq += sum([[L_t(x1.subs(t=1)**k) - Lplus_t(x1.subs(t=0)**k), 
                           L_t(x2.subs(t=1)**k) - Lplus_t(x2.subs(t=0)**k)]
                      for k  in range(1,max_deg+1)],
                         [])


    # make matrices sos on [0, 1]
    for i, Mi in tqdm(enumerate(loc_mom_matrices.values()), total=len(loc_mom_matrices), 
                      desc='make matrices sos on [0, 1]'):
        if t in Mi.variables():
            sos_const = make_matrix_sos_on_0_1(Mi, 'Q'+str(i+1))
            linear_eq += sos_const['linear_eq']
            psd_matrices +=  sos_const['psd_matrices']
        else:
            psd_matrices.append(Mi)


    # decision var
    decision_variables = sum( (list(phi.values())for phi in phis), []) +\
                        sum(map(lambda Q:list(Q.variables()), psd_matrices), [])# +\
                        #length_piece + length_piece_squared        
    decision_variables = [v for v in decision_variables if v != 1]                        
    decision_variables = list(set(decision_variables))


    return Lphis, decision_variables, linear_eq, psd_matrices


def optimize_one_iteration(P, sdp_data, old_solution_exp, max_deg, weight_length_segment, reg_magnitude, solver):
    Lphis, decision_variables, linear_eq, psd_matrices = sdp_data
    t, x1, x2, u1, u2, v1, v2 = P.gens()

    objective = sum(Lphi(u1**max_deg+v1**max_deg+u2**max_deg+v2**max_deg \
                         - max_deg*np.array([u1,u2,v1,v2]).dot(np.power(old_uv_i, max_deg-1))) 
                    for Lphi, old_uv_i in zip(Lphis, old_solution_exp))

    
    sdp_problem = sage_cvxpy.SageCVXProblem(
        list_vars=decision_variables,
        equality_constraints=linear_eq,
        inequality_constraints=[],
        psd_matrices=psd_matrices,
        objective=-objective,
        reg_variables=list(decision_variables), reg_magnitude=reg_magnitude,
        linear_segments=[[Lphi(u2**2), Lphi(v2**2)] for Lphi in Lphis], 
        linear_magnitude=weight_length_segment,
        debug=False)

    solver_params = {'solver':  str.upper(solver), 'verbose': False}
    status, prob_sol = sdp_problem.solve(**solver_params)

    new_solution_exp = [ vector([Lphi(u1), Lphi(u2), Lphi(v1), Lphi(v2),]).subs(**prob_sol) for Lphi in Lphis]
    return new_solution_exp
