#Example ODE fitting

##### Mono culture #####
ssq = function(p){
  t = unique(tmp$Time) #tmp is the data set fitted to
  #solve ode
  out = data.frame(ode(y = c(A = init),times = t,
                       func = genlog,parms=c(p)))
  names(out) = c("Time","A")
  
  ssqres1 = c()
  for(i in 1:nrow(tmp)){
    filt = out %>% filter(Time == tmp$Time[i])
    resA = mean(filt$A)
    ssqres1[i] = (tmp$A[i] - resA)
  }
  ssqres=c(ssqres1)
  return (ssqres)
}
p=c(1,0.1)
mod = nls.lm(par = p, fn = ssq)