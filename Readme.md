# Symbolic matrix differentiation in sympy

What it says on the tin! 

Many thanks to the author of https://zulko.wordpress.com/2012/04/15/symbolic-matrix-differentiation-with-sympy/ for laying out a much earlier version of this I have built on. 


##
setup:

sudo apt-get install python3-sympy

(you can also use python2, but tab completion is nice)

##Example usage:

from symbdiff import diff
a=diff.MatrixSymbol('a',1,5)
print(diff.matDiff(diff.Transpose(a)*a,a))
[a'*d(a) + d(a)'*(a)]

X=diff.MatrixSymbol('X',5,5)
print(diff.matDiff(a*X*diff.Transpose(a),a))

[d(a)*(X*a') + a*X*d(a)' + a*0*(a')]
