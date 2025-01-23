## Hamiltonian du Goddard

Définition: 
$$
\begin{align}
\max \ &f(\bold{u}) \\
 s.t.\ &\bold{g}_k(\bold{x},\bold{u}) = 0 \quad \forall k\\
 & \bold{x_l} \le \bold{x}_k \le\bold{x_u} \quad \forall k\\
 & \bold{x_l} \le \bold{u}_k \le\bold{x_u} \quad \forall k \\
\end{align}
$$
Notations (Euler explicite):
$$
\begin{align}
\bold{x}_k &= [h_k,v_k,m_k]^T \\ 
\bold{u}_k &= [T_k, \Delta]^T \\ 
f(\bold u ) &= h_{\text{fin}} \\ 
\bold{g}_k &= [g_k^1,g_k^2,g_k^3]\\
g_k^1(\bold{x},\bold{k}) &= h_{k+1} - h_{k} - \Delta \times \Phi(\bold{x_k},\bold{u_k})\\
g_k^2(\bold{x},\bold{k}) &= v_{k+1} - v_{k} - \Delta \times \Phi(\bold{x_k},\bold{u_k})\\ 
g_k^3(\bold{x},\bold{k}) &= m_{k+1} - m_{k} - \Delta \times \Phi(\bold{x_k},\bold{u_k})\\
W_k &= f(\bold{u}) - \bold{\lambda}_k^T \bold g_k(\bold{x},\bold{u}) -\alpha_u(x_u-x_k) -\alpha_l(x_k-x_l)-\beta_u(u_u - u_k) - \beta_l(u_k-u_l) 
\end{align}
$$
Et la dynamique:
$$
\Phi(h,v,m,u)= \left[ \begin{align} &v \\ \frac{h_0u-D_cv^2\exp(h_c(h-h_0))}{h_0m} &- \frac{gh_0^2}{h^2} \\ &-\frac{u}{c} \end{align} \right]
$$
Donc 
$$
\begin{align}
\bold{W}_{\bold{xx}} &= -\lambda^T\nabla_{\bold{xx}}\bold{g}(x,u) = -\lambda^T\nabla_{\bold{xx}}\left[ \begin{matrix}h_{k+1} - h_k - v_k\times \Delta \\ v_{k+1} - v_k - (\frac{h_0u-D_cv^2\exp(h_c(h-h_0))}{h_0m} - \frac{gh_0^2}{h^2})\times \Delta \\ m_{k+1} - m_k + \frac{u}{c}\times \Delta \end{matrix} \right] 
\\ 
&=-\bold{\lambda}^T \nabla_\bold{x} \left[\begin{matrix} 1 &-\Delta &0 \\ \frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m} - \frac{\Delta2gh_0^2}{h^3} & 1+\frac{2\Delta D_c v \exp(h_c(h-h_0))}{h_0m} &\frac{\Delta u}{m^2}-\frac{\Delta D_cv^2\exp(h_c(h-h_0))}{h_0m^2}\\ 0&0&1 \end{matrix}\right]
\\
&=- \nabla_\bold{x}\bold{\lambda}^T \left[\begin{matrix} 1 &-\Delta &0 \\ \frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m} - \frac{\Delta2gh_0^2}{h^3} & 1+\frac{2\Delta D_c v \exp(h_c(h-h_0))}{h_0m} &\frac{\Delta u}{m^2}-\frac{\Delta D_cv^2\exp(h_c(h-h_0))}{h_0m^2}\\ 0&0&1 \end{matrix}\right]
\\
&=- \nabla_\bold{x}\left[\begin{matrix} \lambda_h -\Delta\lambda_v \\ \lambda_h(\frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m} - \frac{\Delta2gh_0^2}{h^3}) +\lambda_v( 1+\frac{2\Delta D_c v \exp(h_c(h-h_0))}{h_0m}) + \lambda_m(\frac{\Delta u}{m^2}-\frac{\Delta D_cv^2\exp(h_c(h-h_0))}{h_0m^2}) \\ \lambda_m \end{matrix}\right]^T
\\
&=-\left[\begin{matrix} 0&\lambda_h(\frac{\Delta D_cv^2h_c^2\exp(h_c(h-h_0))}{h_0m}+\frac{6\Delta gh_0^2}{h^4}) +\lambda_v\frac{2\Delta D_cvh_c\exp(h_c(h-h_0))}{h_0m}+\lambda_m(\frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m^2}) & 0\\ 0& \lambda_h\frac{2\Delta D_cvh_c\exp(h_c(h-h_0))}{h_0m}+\lambda_v\frac{2\Delta D_c\exp(h_c(h-h_0))}{h_0m} - \lambda_m\frac{2\Delta D_c v \exp(h_c(h-h_0))}{h_0m^2} & 0\\0 & -\lambda_h\frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m^2} - \lambda_v \frac{2\Delta D_cv\exp(h_c(h-h_0))}{h_0m^2} - \lambda_m\frac{2\Delta u}{m^3}+\lambda_m\frac{2\Delta D_cv^2\exp(h_c(h-h_0))}{h_0m^3}&0  \end{matrix}\right]

\end{align}
$$

$$
\begin{align}
W_{\bold{ux}}
&=- \nabla_\bold{u}\left[\begin{matrix} \lambda_h -\Delta\lambda_v \\ \lambda_h(\frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m} - \frac{\Delta2gh_0^2}{h^3}) +\lambda_v( 1+\frac{2\Delta D_c v \exp(h_c(h-h_0))}{h_0m}) + \lambda_m(\frac{\Delta u}{m^2}-\frac{\Delta D_cv^2\exp(h_c(h-h_0))}{h_0m^2}) \\ \lambda_m \end{matrix}\right]^T
\\
&=-\left[\begin{matrix} \frac{\part}{\part u}\\ \frac{\part}{\part \Delta} \end{matrix}\right] 
\left[\begin{matrix} \lambda_h -\Delta\lambda_v \\ \lambda_h(\frac{\Delta D_cv^2h_c\exp(h_c(h-h_0))}{h_0m} - \frac{\Delta2gh_0^2}{h^3}) +\lambda_v( 1+\frac{2\Delta D_c v \exp(h_c(h-h_0))}{h_0m}) + \lambda_m(\frac{\Delta u}{m^2}-\frac{\Delta D_cv^2\exp(h_c(h-h_0))}{h_0m^2}) \\ \lambda_m \end{matrix}\right]^T
\\
&= -\left[ \begin{matrix}  0 & \frac{\lambda_m \Delta }{m^2}& 0\\ -\lambda_v &\lambda_h(\frac{D_cv^2h_c\exp(hc(h-h_0))}{h_0m}-\frac{2gh_0^2}{h^3}) + \lambda_v\frac{2D_cv\exp(hc(h-h_0))}{h_0m} + \lambda_m(\frac{u}{m^2} - \frac{D_cv^2\exp(h_c(h-h_0))}{h_0m^2})&0 \end{matrix}\right]

\end{align}
$$

