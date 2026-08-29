import numpy as np                                                                                               
from dataclasses import dataclass                                                                                
                                                                                                                
def quintic_coeff(p0: float, v0: float, a0: float,                                                               
                p1: float, v1: float, a1: float, T: float) -> np.ndarray:                                      
    """Solve A*c = b for p(t)=c0+c1 t+...+c5 t^5                                
    BC: p(0)=p0, pdot(0)=v0, pddot(0)=a0, p(T)=p1, pdot(T)=v1, pddot(T)=a1                                    
    """                                                                                                          
    if T < 1e-6:                                                                                                 
        raise ValueError("T too small")                                                                          
    A = np.array([                                                                                               
        [1,0,0,0,0,0],                                                                                           
        [0,1,0,0,0,0],                                                                                           
        [0,0,2,0,0,0],                                                                                           
        [1,T,T**2,T**3,T**4,T**5],                                                                               
        [0,1,2*T,3*T**2,4*T**3,5*T**4],                                                                          
        [0,0,2,6*T,12*T**2,20*T**3],                                                                             
    ], dtype=np.float64)                                                                                         
    b = np.array([p0,v0,a0,p1,v1,a1], dtype=np.float64)                                                          
    c = np.linalg.solve(A, b)                                                                                    
    return c                                                                                                     
                                                                                                                
def quartic_coeff(p0: float, v0: float, a0: float,                                                               
                v1: float, a1: float, T: float) -> np.ndarray:                                                 
    """5 coeffs for braking lon when p1 free: p(t)=c0..c4 """                             
    A = np.array([                                                                                               
        [1,0,0,0,0],                                                                                             
        [0,1,0,0,0],                                                                                             
        [0,0,2,0,0],                                                                                             
        [0,1,2*T,3*T**2,4*T**3],                                                                                 
        [0,0,2,6*T,12*T**2],                                                                                     
    ], dtype=np.float64)                                                                                         
    b = np.array([p0,v0,a0,v1,a1], dtype=np.float64)                                                             
    return np.linalg.solve(A,b)                                                                                  
                                                                                                                
@dataclass                                                                                                       
class Quintic:                                                                                                   
    c: np.ndarray  # (6,)                                                                                        
    T: float                                                                                                     
    def eval(self, t: float) -> float:                                                                           
        c=self.c; return c[0]+c[1]*t+c[2]*t**2+c[3]*t**3+c[4]*t**4+c[5]*t**5                                     
    def eval_d(self, t: float) -> float:                                                                         
        c=self.c; return c[1]+2*c[2]*t+3*c[3]*t**2+4*c[4]*t**3+5*c[5]*t**4                                       
    def eval_dd(self, t: float) -> float:                                                                        
        c=self.c; return 2*c[2]+6*c[3]*t+12*c[4]*t**2+20*c[5]*t**3                                               
    def jerk(self, t: float) -> float:                                                                           
        c=self.c; return 6*c[3]+24*c[4]*t+60*c[5]*t**2                                                           
    def eval_vec(self, ts: np.ndarray) -> np.ndarray:                                                            
        c=self.c                                                                                                 
        return c[0]+c[1]*ts+c[2]*ts**2+c[3]*ts**3+c[4]*ts**4+c[5]*ts**5                                          

# no use  
@dataclass                                                                                                       
class Quartic:                                                                                                   
    c: np.ndarray  # (5,)                                                                                        
    T: float                                                                                                     
    def eval(self,t:float)->float:                                                                               
        c=self.c; return c[0]+c[1]*t+c[2]*t**2+c[3]*t**3+c[4]*t**4                                               
    def eval_d(self,t:float)->float:                                                                             
        c=self.c; return c[1]+2*c[2]*t+3*c[3]*t**2+4*c[4]*t**3                                                   
    def eval_dd(self,t:float)->float:                                                                            
        c=self.c; return 2*c[2]+6*c[3]*t+12*c[4]*t**2                                                            
                                                                                                                
def lateral_quintic(d0, d_dot0, d_target, T):                                                                    
    return Quintic(quintic_coeff(d0, d_dot0, 0.0, d_target, 0.0, 0.0, T), T)                                     
                                                                                                                
def longitudinal_quintic(s0, v0, s_target, v_target, T):                                                         
    return Quintic(quintic_coeff(s0, v0, 0.0, s_target, v_target, 0.0, T), T) 