%% VRS Input Oriented Envelopment form
% function [e_Input_VRS_Envelopment,optimal_solution] = InputOriented_VRS_Envelopment(Data,InputsNumber,OutputsNumber)
function [e_Input_VRS_Envelopment,Iterations] = InputOriented_VRS_Envelopment(Data,InputsNumber,OutputsNumber)
m = InputsNumber+OutputsNumber;
n = size(Data,1);
e_Input_VRS_Envelopment = zeros(n,1);
optimal_solution = zeros(n+1,n);

options = optimoptions('linprog','Algorithm','dual-simplex','Display','off');
% options = optimoptions('linprog','Algorithm','dual-simplex','Preprocess','none');
for i=1:n
    f = [zeros(1,n) 1 ]; % min θ (Objective function)
    lx = [Data(:,1:InputsNumber)' -Data(i,1:InputsNumber)'];  % λX -θXo <= 0
    ly = [-Data(:,InputsNumber+1:m)' zeros(1,OutputsNumber)']; % -λY <= -Yo
    sl = [ones(1,n) 0]; % Σλ = 1
    
    A = [lx;ly]; % LHS of Inequality Constraints of type <=
    b = [zeros(1,InputsNumber) -Data(i,InputsNumber+1:m)]; % RHS of Inequality Constraints
    Aeq = sl; % LHS of Equality Constraints
    beq = 1;  % RHS of Equality Constraints
    
    lb=[zeros(1,n) -inf]; % Lower bounds of the variables
    
    % Call the solver and save the optimal solution, the optimal value of the objective function and information about the optimization process
    [opt_sol,fval,exitflag,output] = linprog(f,A,b,Aeq,beq,lb,[],options);
    Iterations(i,1) = output.iterations;
    e_Input_VRS_Envelopment(i,1) = fval; % Save the efficiency score of each unit
    optimal_solution(:,i) = opt_sol; % Save the optimal solution of each unit       
end

end