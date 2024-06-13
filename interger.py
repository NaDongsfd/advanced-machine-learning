from sympy import *
x = symbols('x')
print(integrate(x, (x, 1, 2)))

#import sympy
#from sympy.abc import x  # 使用符号变量的时候，需要先导入符号
#value = sympy.integrate(sympy.sin(x)*sympy.cos(x), (x, 0, sympy.pi/2))
