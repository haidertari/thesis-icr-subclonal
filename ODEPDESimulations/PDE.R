##### Mono-culture model #####
derivativesFuncPDE = function(A,B,dx){
  
  max = length(A)
  
  Aleft   = A[1:(max-2)]
  Aright  = A[3:(max)]
  Acenter = A[2:(max-1)]
  
  Bleft   = B[1:(max-2)]
  Bright  = B[3:(max)]
  Bcenter = B[2:(max-1)]
  
  tmp1 = ((Aright - Aleft) * Bcenter * (Bright - Bleft))
  tmp2 = (4 * Acenter * Bcenter * (Bright - 2 * Bcenter + Bleft))
  tmp3 = Acenter * ((Bright - Bleft)^2)
  
  Q = (tmp1+tmp2-tmp3)/(4*dx*Bcenter**2)
  
  return(Q)
  
}
runmonoPDEModel = function(growth = 1, death = 0, advection = 1,
                           radius = 90, gridSize = 1000){
  
  dx = 1.0/gridSize      # space step
  dt = 0.0000001            # time step
  Ts = 6.0               # total time
  n = ceiling(Ts/dt)+1   # number of iterations
  index = 1              # indexing point
  result = c()           # save results here
  simtime = c()          # save simulation time here
  step_write = floor(n / Ts)
  # set up species vector
  x = vector(mode="numeric",length=1000)
  
  x[1:radius] = 1
  x[(radius+1):gridSize] = exp( -0.1 * (1:(gridSize-radius)) )
  x[1:radius] = 1
  x[(radius+1):gridSize] = exp( -0.1 * (1:(gridSize-radius)) )
  
  res = c()
  
  for(i in 0:n){
    
    if(i %% step_write == 0){
      res[index] = sum(x)
      index = index + 1
    }
    
    x[1] = x[2]
    x[gridSize] = x[gridSize-1]
    
    # compute derivatives
    delta_x = derivativesFuncPDE(x,x,dx)
    
    # take points not in boundary
    xc = x[2:(gridSize-1)]
    
    #update variables
    x[2:(gridSize-1)] = xc + dt * ( ( advection * delta_x ) +
                                      ( growth * xc * (1 - xc) ) -
                                      ( death * xc ) )
  }
  return(res)
}

##### Co-culture Model #####
runcoPDEModel = function(growth = c(1,1), interaction = c(1,1), death_mod1 = c(0,0), death_mod2 = c(0,0), advection = c(1,1),
                         radius = 80, gridSize = 1000, ratio = 0.5){
  
  dx = 1.0/gridSize      # space step
  dt = 0.000001            # time step
  Ts = 6.0               # total time
  n = ceiling(Ts/dt)+1   # number of iterations
  index = 1              # indexing point
  result = c()           # save results here
  simtime = c()          # save simulation time here
  step_write = floor(n / Ts)
  # set up species vector
  x = vector(mode="numeric",length=1000)
  y = vector(mode="numeric",length=1000)
  
  x[1:radius] = ratio
  x[(radius+1):gridSize] = (ratio)*(exp( -1 * (1:(gridSize-radius)) ))
  y[1:radius] = 1-ratio
  y[(radius+1):gridSize] = (1-ratio)*(exp( -1 * (1:(gridSize-radius)) ))
  x[x<0.00002]=0.00001
  y[y<0.00002]=0.00001
  res_x = c()
  res_y = c()
  
  for(i in 0:n){
    
    #This prevents dividing by zero in calculating the flux
    if(i %% step_write == 0){
      res_x[index] = sum(x[x>0.00002])
      res_y[index] = sum(y[y>0.00002])
      index = index + 1
    }
    x[x<0.00002]=0.00001
    y[y<0.00002]=0.00001
    x[1] = x[2]
    x[gridSize] = x[gridSize-1]
    y[1] = y[2]
    y[gridSize] = y[gridSize-1]
    
    # compute derivatives
    delta_x = derivativesFuncPDE(x,x+y,dx)
    delta_y = derivativesFuncPDE(y,x+y,dx)
    
    # take points not in boundary
    xc = x[2:(gridSize-1)]
    yc = y[2:(gridSize-1)]
    
    #update variables
    x[2:(gridSize-1)] = xc + dt * ( ( advection[1] * delta_x ) +
                                      ( (growth[1]-death_mod2[1]) * xc * (1 - xc - (interaction[1]*yc)) ) -
                                      ( death_mod1[1] * xc ) )
    y[2:(gridSize-1)] = yc + dt * ( ( advection[2] * delta_y ) +
                                      ( (growth[2]-death_mod2[2]) * yc * (1 - (interaction[2]*xc) - yc) ) -
                                      ( death_mod1[2] * yc ) )
  }
  times = seq(0,6,1)

  
  return(list(res_x,res_y))
  
}
