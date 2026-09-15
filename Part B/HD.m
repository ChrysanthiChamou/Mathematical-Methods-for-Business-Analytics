function [e_Input_VRS, ...
          Efficient_ids, ...
          Inefficient_ids, ...
          Time_Phase1, ...
          Time_Phase2] = ...
    HD(Data,InputsNumber,OutputsNumber,...
       beta,gamma,block_size,useParallel)

%% =========================================================
% HIERARCHICAL DECOMPOSITION - HD
% Input Oriented DEA - VRS
% =========================================================

[n,m] = size(Data);

%% =========================================================
% PHASE I
% =========================================================

fprintf('\nStarting HD Phase I...\n');

start_time_HD_Phase1 = tic;

% Initially all DMUs are unknown
All_ids = (1:n)';

%% Level 1

[Unknown_Status_Block_ids,~] = ...
    SolveBlocks( ...
    Data,...
    InputsNumber,...
    OutputsNumber,...
    All_ids,...
    block_size);

U_size1 = length(Unknown_Status_Block_ids);

fprintf('Level 1: %d candidate DMUs\n',U_size1);

%% Levels 2+

while U_size1 > 0

    [New_Unknown_ids,~] = ...
        SolveBlocks( ...
        Data,...
        InputsNumber,...
        OutputsNumber,...
        Unknown_Status_Block_ids,...
        block_size);

    U_size2 = length(New_Unknown_ids);

    if U_size1 > 0
        ratio = U_size2 / U_size1;
    else
        ratio = 0;
    end

    fprintf('Next level: %d candidates, ratio = %.4f\n',...
        U_size2,ratio);

    %% Stopping condition

    if U_size2 == 0

        % No candidates remain
        Unknown_Status_Block_ids = [];

        break;

    elseif ratio > gamma

        fprintf('HD stopping condition reached.\n');

        %% Final DEA on remaining candidates

        [e_temp,~] = ...
            InputOriented_VRS_Envelopment( ...
            Data(New_Unknown_ids,:),...
            InputsNumber,...
            OutputsNumber);

        temp_ids = ...
            New_Unknown_ids(e_temp >= 0.9999);

        Unknown_Status_Block_ids = New_Unknown_ids;

        Efficient_ids = ...
            temp_ids(:);

        break;

    else

        % Continue to next level
        Unknown_Status_Block_ids = New_Unknown_ids;

        block_size = ceil(beta * block_size);

        U_size1 = U_size2;

    end

end

%% =========================================================
% SAFETY CHECK / FINAL EVALUATION
% =========================================================

if ~exist('Efficient_ids','var') || isempty(Efficient_ids)

    if ~isempty(Unknown_Status_Block_ids)

        fprintf('Final DEA evaluation...\n');

        [e_temp,~] = ...
            InputOriented_VRS_Envelopment( ...
            Data(Unknown_Status_Block_ids,:),...
            InputsNumber,...
            OutputsNumber);

        Efficient_ids = ...
            Unknown_Status_Block_ids( ...
            e_temp >= 0.9999);

    else

        Efficient_ids = [];

    end

end

%% Unique and sorted IDs

Efficient_ids = ...
    sort(unique(Efficient_ids(:)));

%% Inefficient IDs

Inefficient_ids = ...
    All_ids(~ismember(All_ids,Efficient_ids));

%% Phase I time

Time_Phase1 = toc(start_time_HD_Phase1);

fprintf('\n-----------------------------------------\n');
fprintf('HD PHASE I FINISHED\n');
fprintf('Efficient DMUs   : %d\n',length(Efficient_ids));
fprintf('Inefficient DMUs : %d\n',length(Inefficient_ids));
fprintf('Phase I time     : %.4f sec\n',...
    Time_Phase1);
fprintf('-----------------------------------------\n');


%% =========================================================
% PHASE II
% =========================================================

fprintf('\nStarting HD Phase II...\n');

start_time_HD_Phase2 = tic;

e_Input_VRS = ones(n,1);

Efficient_Size = length(Efficient_ids);
Inefficient_Size = length(Inefficient_ids);

if Efficient_Size == 0

    error('HD produced zero efficient DMUs.');

end

%% =========================================================
% PARALLEL PHASE II
% =========================================================

if useParallel

    fprintf('Phase II: PARFOR\n');

    PhaseII_scores = ones(Inefficient_Size,1);

    parfor k = 1:Inefficient_Size

        dmu = Inefficient_ids(k);

        PhaseII_scores(k) = ...
            SolveOneDMU( ...
            Data,...
            InputsNumber,...
            OutputsNumber,...
            Efficient_ids,...
            dmu);

    end

    e_Input_VRS(Inefficient_ids) = ...
        PhaseII_scores;

%% =========================================================
% SERIAL PHASE II
% =========================================================

else

    fprintf('Phase II: FOR\n');

    for k = 1:Inefficient_Size

        dmu = Inefficient_ids(k);

        e_Input_VRS(dmu) = ...
            SolveOneDMU( ...
            Data,...
            InputsNumber,...
            OutputsNumber,...
            Efficient_ids,...
            dmu);

    end

end

%% Efficient DMUs

e_Input_VRS(Efficient_ids) = 1;

%% Phase II time

Time_Phase2 = toc(start_time_HD_Phase2);

fprintf('\n-----------------------------------------\n');
fprintf('HD PHASE II FINISHED\n');
fprintf('Phase II time     : %.4f sec\n',...
    Time_Phase2);
fprintf('-----------------------------------------\n');

end


%% =========================================================
% SOLVE ONE DMU
% =========================================================

function score = ...
    SolveOneDMU( ...
    Data,...
    InputsNumber,...
    OutputsNumber,...
    Efficient_ids,...
    DMU_id)

m = InputsNumber + OutputsNumber;

Efficient_Size = length(Efficient_ids);

%% Objective

f = [zeros(1,Efficient_Size) 1];

%% Input constraints

lx = [ ...
    Data(Efficient_ids,1:InputsNumber)' ...
    -Data(DMU_id,1:InputsNumber)' ];

%% Output constraints

ly = [ ...
    -Data(Efficient_ids,InputsNumber+1:m)' ...
    zeros(OutputsNumber,1) ];

%% Constraints

A = [lx;ly];

b = [ ...
    zeros(1,InputsNumber) ...
    -Data(DMU_id,InputsNumber+1:m)];

%% VRS constraint

Aeq = [ones(1,Efficient_Size) 0];

beq = 1;

%% Bounds

lb = [zeros(1,Efficient_Size) -inf];

%% MATLAB R2025b options

options = optimoptions( ...
    'linprog',...
    'Algorithm','dual-simplex',...
    'Display','off');

%% Solve LP

[~,fval,exitflag] = ...
    linprog( ...
    f,...
    A,b,...
    Aeq,beq,...
    lb,[],...
    options);

if exitflag > 0

    score = fval;

else

    score = NaN;

end

end