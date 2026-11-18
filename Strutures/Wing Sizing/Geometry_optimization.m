clear all
clc


M_root=16.9 * 12; %lbf*in
w=10;
w_range=linspace(5,20,100);
h=.738;
h_range = linspace(0.3,0.738,100);
c=h/2;

t1=.04;
t2=.02;
t1_range=linspace(0.010,0.100,100);
t2_range=linspace(0.010,0.050,100);

I = @(t1,t2,w,h) 2*(1/12).*(t1).*(h-2.*t2).^3 + 2*((1/12).*(w).*(t2).^3+(w.*t2).*((h./2)-(t2)./2).^2);
A = @(t1,t2,w,h) 2.*t1.*(h-2.*t2) + 2.*w.*t2;

%Range t1
I_t1range=I(t1_range,t2,w,h);
stress_t1range=M_root*c./I_t1range;
A_t1range=A(t1_range,t2,w,h);

figure
plot(t1_range,stress_t1range,'LineWidth',2)
xlabel('Spar Thickness (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Spar Thickness')
grid on

figure
plot(A_t1range,stress_t1range,'LineWidth',2)
xlabel('Cross-Sectional Area (in^2)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Cross-Sectional Area, varied spar thickness')
grid on

%Range t2
I_t2range=I(t1,t2_range,w,h);
stress_t2range=M_root*c./I_t2range;
A_t2range=A(t1,t2_range,w,h);

figure
plot(t2_range,stress_t2range,'LineWidth',2)
xlabel('Skin Thickness (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Skin Thickness')
grid on

figure
plot(A_t2range,stress_t2range,'LineWidth',2)
xlabel('Cross-Sectional Area (in^2)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Cross-Sectional Area, varied skin thickness')
grid on

%Range w
I_wrange=I(t1,t2,w_range,h);
stress_wrange=M_root*c./I_wrange;
A_wrange=A(t1,t2,w_range,h);

figure
plot(w_range,stress_wrange,'LineWidth',2)
xlabel('Box Width (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Box Width')
grid on

figure
plot(A_wrange,stress_wrange,'LineWidth',2)
xlabel('Cross-Sectional Area (in^2)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Cross-Sectional Area, varied box width')
grid on

% Range h
I_hrange = I(t1,t2,w,h_range);
stress_hrange = M_root.*(h_range./2)./I_hrange;
A_hrange = A(t1,t2,w,h_range);

figure
plot(h_range,stress_hrange,'LineWidth',2)
xlabel('Box Height (in)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Box Height')
grid on

figure
plot(A_hrange,stress_hrange,'LineWidth',2)
xlabel('Cross-Sectional Area (in^2)')
ylabel('Bending Stress (psi)')
title('Bending Stress vs Cross-Sectional Area, varied box height')
grid on


% Baseline bending stress
I_base = I(t1,t2,w,h);
stress_base = M_root*c/I_base;

% Increase each variable by 10%
stress_t1_10 = M_root*c/I(t1*1.10,t2,w,h);
stress_t2_10 = M_root*c/I(t1,t2*1.10,w,h);
stress_w_10  = M_root*c/I(t1,t2,w*1.10,h);

% Height changes both I and c
h_10 = h*1.10;
stress_h_10 = M_root*(h_10/2)/I(t1,t2,w,h_10);

% Calculate percentage reduction in stress
reduction_t1_10 = 100*(stress_base-stress_t1_10)/stress_base;
reduction_t2_10 = 100*(stress_base-stress_t2_10)/stress_base;
reduction_w_10  = 100*(stress_base-stress_w_10)/stress_base;
reduction_h_10  = 100*(stress_base-stress_h_10)/stress_base;

% Create bar chart
reductions_10 = [reduction_t1_10, reduction_t2_10, ...
                 reduction_w_10, reduction_h_10];

figure
bar(reductions_10)

set(gca,'XTick',1:4, ...
    'XTickLabel',{'Spar Thickness','Skin Thickness', ...
                  'Box Width','Box Height'})

ylabel('Bending Stress Reduction (%)')
title('Bending Stress Sensitivity to a 10% Increase in Each Variable')
grid on

% Display percentages above bars
for i = 1:length(reductions_10)
    text(i, reductions_10(i)+0.3, ...
        sprintf('%.2f%%',reductions_10(i)), ...
        'HorizontalAlignment','center', ...
        'FontSize',12, ...
        'FontWeight','bold')
end

ylim([0 max(reductions_10)*1.2])
