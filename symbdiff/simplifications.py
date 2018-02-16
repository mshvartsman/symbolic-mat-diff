from sympy import Trace, MatMul, preorder_traversal, MatAdd, Add
from collections import OrderedDict


def simplify_matdiff(expr, dX):
	for cond, repl in rules.items():
		expr = _conditional_replace(expr, cond(dX), repl(dX))
	return expr


# simplification rules
def _cyclic_permute(expr):
	if expr.is_Trace and expr.arg.is_MatMul:
		prods = expr.arg.args
		newprods = [prods[-1], *prods[:-1]]
		return Trace(MatMul(*newprods))
	else:
		print(expr)
		raise RuntimeError("Only know how to cyclic permute products inside traces!")


def _conditional_replace(expr, condition, replacement):
	for x in preorder_traversal(expr):
		try:
			if condition(x):
				expr = expr.xreplace({x: replacement(x)})
		except AttributeError:  # scalar ops like Add won't have is_Trace
			pass
	return expr


# conditions
def transpose_traces_cond(dX):
	def cond(x):
		return x.is_Trace and x.arg.is_MatMul and x.has(dX.T)
	return cond


def transpose_traces_repl(dX):
	return lambda x: Trace(x.arg.T)


def trace_sum_distribute_cond(dX):
	return lambda x: x.is_Trace and x.arg.is_MatAdd


def trace_sum_distribute_repl(dX):
	return lambda x: Add(*[Trace(A) for A in x.arg.args])


def matmul_distribute_cond(dX):
	return lambda x: x.is_MatMul and x.has(MatAdd)


def matmul_distribute_repl(dX):
	def repl(x):
		pre, post = [], []
		sawAdd = False
		for arg in x.args:
			if arg.is_MatAdd:
				sawAdd = True
				add = arg
				continue
			if not sawAdd:
				pre.append(arg)
			else:
				post.append(arg)
		# ugly hack here because I can't figure out how to not end up 
		# with nested parens that break other things
		try:
			addends = [[*addend.args] if addend.is_MatMul else [addend] for addend in add.args]
			return MatAdd(*[MatMul(*[*pre, *addend, *post]) for addend in addends])
		except UnboundLocalError:
			return x
	return repl


def inverse_transpose_cond(dX):
	return lambda x: x.is_Transpose and x.arg.is_Inverse and x.arg.is_Symmetric


def inverse_transpose_repl(dX):
	return lambda x: x.arg


def cyclic_permute_dX_cond(dX):
	def cond(x):
		return x.is_Trace and x.has(dX) and x.arg.args[-1] != dX
	return cond


def cyclic_permute_dX_repl(dX):
	def repl(x):
		newx = x
		nperm = 0
		while newx.arg.args[-1] != dX:
			newx = _cyclic_permute(newx)
			nperm = nperm + 1
			if nperm > len(newx.arg.args):
				raise RuntimeError("Cyclic permutation failed to move dX to end!")
		return newx
	return repl

# rules applied in order, this should hopefully work to simplify
rules = OrderedDict([(matmul_distribute_cond, matmul_distribute_repl),
					(trace_sum_distribute_cond, trace_sum_distribute_repl),
					(transpose_traces_cond, transpose_traces_repl),
					(inverse_transpose_cond, inverse_transpose_repl),
					(cyclic_permute_dX_cond, cyclic_permute_dX_repl)])
