clear all
clc


M_root=16.9 * 12; %lbf*in
w=10;
w_range=linspace(5,20,100);
h=.738;
c=h/2;

t1=.04;
t2=.02;
t1_range=linspace(0.010,0.100,100);
t2_range=linspace(0.010,0.050,100);

I = @(t1,t2,w,h) 2*(1/12).*(t1).*(h-2.*t2).^3 + 2*((1/12).*(w).*(t2).^3+(w.*t2).*((h./2)-(t2)./2).^2);

%Range t1
I_t1range=I(t1_range,t2,w,h);
stress_t1range=M_root*c./I_t1range;

figure
plot(t1_range,stress_t1range,'LineWidth',2)
xlabel('Spar Thickness (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Spar Thickness')
grid on

%Range t2
I_t2range=I(t1,t2_range,w,h);
stress_t2range=M_root*c./I_t2range;

figure
plot(t2_range,stress_t2range,'LineWidth',2)
xlabel('Skin Thickness (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Skin Thickness')
grid on

%Range w
I_wrange=I(t1,t2,w_range,h);
stress_wrange=M_root*c./I_wrange;

figure
plot(w_range,stress_wrange,'LineWidth',2)
xlabel('Box Width (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Box Width')
grid on


% Derivative of stress with respect to skin thickness
dstress_dt2 = gradient(stress_t2range, t2_range);

figure
plot(t2_range, dstress_dt2, 'LineWidth', 2)
xlabel('Skin Thickness (in)')
ylabel('dStress/dt2 (psi/in)')
title('Rate of Change of Bending Stress vs Skin Thickness')
grid on


