#! /usr/bin/env python3
#
def rk4 ( dydt, tspan, y0, n ):

#*****************************************************************************80
#
## RK4 approximates the solution to an ODE using the RK4 method.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    function dydt: points to a function that evaluates the right
#    hand side of the ODE.
#
#    real tspan[2]: contains the initial and final times.
#
#    real y0[m]: an array containing the initial condition.
#
#    integer n: the number of steps to take.
#
#  Output:
#
#    real t[n+1], y[n+1,m]: the times and solution values.
#
  import numpy as np

  if ( np.ndim ( y0 ) == 0 ):
    m = 1
  else:
    m = len ( y0 )

  tfirst = tspan[0]
  tlast = tspan[1]
  dt = ( tlast - tfirst ) / n
  t = np.zeros ( n + 1 , dtype=np.single)
  y = np.zeros ( [ n + 1, m ], dtype=np.single)
  t[0] = tspan[0]
  y[0,:] = y0

  for i in range ( 0, n ):

    f1 = dydt ( t[i],            y[i,:] )
    f2 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f1 / 2.0 )
    f3 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f2 / 2.0 )
    f4 = dydt ( t[i] + dt,       y[i,:] + dt * f3 )

    t[i+1] = t[i] + dt
    y[i+1,:] = y[i,:] + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

  return t, y

def rk4_test ( ):

#*****************************************************************************80
#
## rk4_test tests rk4.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'rk4_test:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test rk4.' )

  tspan = np.array ( [ 0.0, 2.0 ] )
  y0 = 5.1765
  n = 100
  rk4_humps_test ( tspan, y0, n )

  tspan = np.array ( [ 0.0, 5.0 ] )
  y0 = np.array ( [ 5000, 100 ] )
  n = 200
  rk4_predator_prey_test ( tspan, y0, n )
#
#  Terminate.
#
  print ( '' )
  print ( 'rk4_test:' )
  print ( '  Normal end of execution.' )
  return

def rk4_humps_test ( tspan, y0, n ):

#*****************************************************************************80
#
## rk4_humps_test solves the humps ODE using the RK4 method.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real tspan[2]: the time span
#
#    real y0[2]: the initial condition.
#
#    integer n: the number of steps to take.
#
  import matplotlib.pyplot as plt
  import numpy as np

  print ( '' )
  print ( 'rk4_humps_test:' )
  print ( '  Solve the humps ODE system using rk4().' )

  t, y = rk4 ( humps_deriv, tspan, y0, n )

  plt.clf ( )

  plt.plot ( t, y, 'r-', linewidth = 2 )

  a = tspan[0]
  b = tspan[1]
  if ( a <= 0.0 and 0.0 <= b ):
    plt.plot ( [a,b], [0,0], 'k-', linewidth = 2 )

  ymin = min ( y )
  ymax = max ( y )
  if ( ymin <= 0.0 and 0.0 <= ymax ):
    plt.plot ( [0, 0], [ymin,ymax], 'k-', linewidth = 2 )

  plt.grid ( True )
  plt.xlabel ( '<--- T --->' )
  plt.ylabel ( '<--- Y(T) --->' )
  plt.title ( 'humps ODE solved by rk4()' )

  filename = 'rk4_humps_test.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )

  return

def rk4_predator_prey_test ( tspan, y0, n ):

#*****************************************************************************80
#
## rk4_predator_prey_test solves the predator prey ODE using the rk4 method.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real tspan[2]: the time span
#
#    real y0[2]: the initial condition.
#
#    integer n: the number of steps to take.
#
  import matplotlib.pyplot as plt
  import numpy as np

  print ( '' )
  print ( 'rk4_predator_prey_test:' )
  print ( '  Solve the predator prey ODE system using rk4().' )

  t, y = rk4 ( predator_prey_deriv, tspan, y0, n )

  plt.clf ( )

  plt.plot ( y[:,0], y[:,1], 'r-', linewidth = 2 )

  plt.grid ( True )
  plt.xlabel ( '<--- Prey --->' )
  plt.ylabel ( '<--- Predators --->' )
  plt.title ( 'predator prey ODE solved by rk4()' )

  filename = 'rk4_predator_prey_test.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )

  return

def humps_deriv ( x, y ):

#*****************************************************************************80
#
## humps_deriv evaluates the derivative of the humps function.
#
#  Discussion:
#
#    y = 1.0 ./ ( ( x - 0.3 )**2 + 0.01 ) \
#      + 1.0 ./ ( ( x - 0.9 )**2 + 0.04 ) \
#      - 6.0
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real x[:], y[:]: the arguments.
#
#  Output:
#
#    real yp[:]: the value of the derivative at x.
#
  yp = - 2.0 * ( x - 0.3 ) / ( ( x - 0.3 )**2 + 0.01 )**2 \
       - 2.0 * ( x - 0.9 ) / ( ( x - 0.9 )**2 + 0.04 )**2

  return yp

def predator_prey_deriv ( t, rf ):

#*****************************************************************************80
#
## predator_prey_deriv evaluates the right hand side of the system.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    George Lindfield, John Penny,
#    Numerical Methods Using MATLAB,
#    Second Edition,
#    Prentice Hall, 1999,
#    ISBN: 0-13-012641-1,
#    LC: QA297.P45.
#
#  Input:
#
#    real T, the current time.
#
#    real RF[2], the current solution variables, rabbits and foxes.
#
#  Output:
#
#    real DRFDT[2], the right hand side of the 2 ODE's.
#
  import numpy as np

  r = rf[0]
  f = rf[1]

  drdt =    2.0 * r - 0.001 * r * f
  dfdt = - 10.0 * f + 0.002 * r * f

  drfdt = np.array ( [ drdt, dfdt ] )

  return drfdt

def timestamp ( ):

#*****************************************************************************80
#
## TIMESTAMP prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    None
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return

if ( __name__ == '__main__' ):
  timestamp ( )
  rk4_test ( )
  timestamp ( )
