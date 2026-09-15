%% main_HD_experiments.m
clear;
clc;
close all;

%% PARAMETERS

AM_SUM = 12;       

beta = 1.5;
gamma = 0.65;
initial_block_size = 100;

cardinalities = [1500 3000 5000 7500];
dimensions = [3 4 5];

useParallel = false;

outputDir = "results";

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

%% PARALLEL POOL

if useParallel
    try
        p = gcp('nocreate');

        if isempty(p)
            parpool('threads');
        end

    catch
        warning('Δεν ήταν δυνατή η χρήση parallel processing.');
        useParallel = false;
    end
end

%% RESULTS TABLE

Results = table();

experiment = 0;

%% 12 EXPERIMENTS

for c = 1:length(cardinalities)

    N = cardinalities(c);

    for d = 1:length(dimensions)

        Dim = dimensions(d);

        experiment = experiment + 1;

        fprintf('\n============================================\n');
        fprintf('Experiment %d / 12\n',experiment);
        fprintf('Cardinality = %d\n',N);
        fprintf('Inputs      = %d\n',Dim);
        fprintf('Outputs     = %d\n',Dim);
        fprintf('============================================\n');

        %% RANDOM SEED

        rng(1000 + experiment,'twister');

        %% DATA GENERATION

        Data = GenerateData(N,Dim,AM_SUM);

        save(fullfile(outputDir, ...
            sprintf('Data_N%d_D%d.mat',N,Dim)), ...
            'Data','N','Dim','AM_SUM');

        %% STANDARD DEA

        fprintf('Running Standard DEA...\n');

        start_standard = tic;

        [eff_standard,~] = ...
            InputOriented_VRS_Envelopment( ...
            Data,Dim,Dim);

        Time_Standard = toc(start_standard);

        Standard_Efficient_ids = ...
            find(eff_standard >= 0.9999);

        %% HD

        fprintf('Running HD...\n');

        [eff_HD, ...
            Efficient_ids, ...
            Inefficient_ids, ...
            Time_Phase1, ...
            Time_Phase2] = ...
            HD(Data,Dim,Dim,...
            beta,gamma,...
            initial_block_size,...
            useParallel);

        %% CHECK

        Same_Efficient_Set = ...
            isequal( ...
            sort(Standard_Efficient_ids(:)), ...
            sort(Efficient_ids(:)));

        %% SAVE RESULTS

        Total_HD_Time = Time_Phase1 + Time_Phase2;

        NewRow = table( ...
            N,...
            Dim,...
            Dim,...
            length(Efficient_ids),...
            length(Inefficient_ids),...
            Time_Phase1,...
            Time_Phase2,...
            Total_HD_Time,...
            Time_Standard,...
            Same_Efficient_Set,...
            'VariableNames',{ ...
            'Cardinality',...
            'Dimension',...
            'Outputs',...
            'EfficientUnits',...
            'InefficientUnits',...
            'Time_Phase1',...
            'Time_Phase2',...
            'Time_HD_Total',...
            'Time_Standard_DEA',...
            'HD_Matches_DEA'});

        Results = [Results;NewRow];

        %% SAVE INDIVIDUAL EXPERIMENT

        save(fullfile(outputDir,...
            sprintf('Run_N%d_D%d.mat',N,Dim)),...
            'Data',...
            'eff_standard',...
            'eff_HD',...
            'Standard_Efficient_ids',...
            'Efficient_ids',...
            'Inefficient_ids',...
            'Time_Phase1',...
            'Time_Phase2',...
            'Time_Standard',...
            'Same_Efficient_Set');

        %% DISPLAY

        fprintf('\nPhase I  = %.4f sec\n',Time_Phase1);
        fprintf('Phase II = %.4f sec\n',Time_Phase2);
        fprintf('Total HD = %.4f sec\n',Total_HD_Time);

        fprintf('Efficient DMUs = %d\n',...
            length(Efficient_ids));

        fprintf('HD matches DEA = %d\n',...
            Same_Efficient_Set);

    end
end

%% SAVE TABLE

writetable(Results,...
    fullfile(outputDir,'HD_results.csv'));

save(fullfile(outputDir,'HD_results.mat'),...
    'Results');

disp(' ');
disp('============================================');
disp('FINAL RESULTS');
disp('============================================');

disp(Results);


%% =====================================================
% LOGISTIC REGRESSION
% IDENTIFICATION OF EFFICIENT DMUs
% =====================================================

fprintf('\n============================================\n');
fprintf('LOGISTIC REGRESSION ANALYSIS\n');
fprintf('============================================\n');

%% SELECT DATASET

Selected_N = 5000;
Selected_D = 4;

%% LOAD SELECTED EXPERIMENT

load(fullfile(outputDir,...
    sprintf('Run_N%d_D%d.mat',...
    Selected_N,Selected_D)),...
    'Data',...
    'Efficient_ids',...
    'Inefficient_ids');

fprintf('\nSelected dataset:\n');
fprintf('Cardinality = %d\n',Selected_N);
fprintf('Dimension   = %d\n',Selected_D);

%% -----------------------------------------------------
% CREATE BINARY TARGET VARIABLE
% -----------------------------------------------------

N_selected = size(Data,1);

% 0 = Inefficient
% 1 = Efficient

Efficiency_Class = zeros(N_selected,1);

Efficiency_Class(Efficient_ids) = 1;

fprintf('\nClass distribution:\n');
fprintf('Efficient DMUs   = %d\n',...
    sum(Efficiency_Class == 1));

fprintf('Inefficient DMUs = %d\n',...
    sum(Efficiency_Class == 0));

%% -----------------------------------------------------
% PREDICTOR VARIABLES
% -----------------------------------------------------

% All inputs and outputs are used as predictors

X_LR = Data;

Y_LR = Efficiency_Class;

%% -----------------------------------------------------
% TRAIN / TEST SPLIT
% -----------------------------------------------------

rng(2026,'twister');

cv = cvpartition(Y_LR,'HoldOut',0.30);

Train_idx = training(cv);
Test_idx  = test(cv);

X_train = X_LR(Train_idx,:);
Y_train = Y_LR(Train_idx);

X_test = X_LR(Test_idx,:);
Y_test = Y_LR(Test_idx);

fprintf('\nTraining observations = %d\n',...
    length(Y_train));

fprintf('Testing observations  = %d\n',...
    length(Y_test));

%% -----------------------------------------------------
% STANDARDIZATION
% -----------------------------------------------------

fprintf('\nStandardizing predictor variables...\n');

% Mean of training set
mu = mean(X_train,1);

% Standard deviation of training set
sigma = std(X_train,[],1);

% Protection against zero standard deviation
sigma(sigma == 0) = 1;

% Standardize training set
X_train_std = (X_train - mu) ./ sigma;

% Standardize test set
% IMPORTANT: use training mean and std
X_test_std = (X_test - mu) ./ sigma;

%% -----------------------------------------------------
% LOGISTIC REGRESSION MODEL
% -----------------------------------------------------

fprintf('\nTraining Logistic Regression model...\n');

Logistic_Model = fitclinear(...
    X_train_std,...
    Y_train,...
    'Learner','logistic',...
    'Regularization','ridge',...
    'Lambda',1e-4,...
    'ClassNames',[0 1]);

%% -----------------------------------------------------
% PREDICTION
% -----------------------------------------------------

[Y_pred,Score] = predict(...
    Logistic_Model,...
    X_test_std);

%% -----------------------------------------------------
% CONFUSION MATRIX
% -----------------------------------------------------

Confusion_Matrix = confusionmat(...
    Y_test,...
    Y_pred,...
    'Order',[0 1]);

TN = Confusion_Matrix(1,1);
FP = Confusion_Matrix(1,2);
FN = Confusion_Matrix(2,1);
TP = Confusion_Matrix(2,2);

%% -----------------------------------------------------
% PERFORMANCE METRICS
% -----------------------------------------------------

Accuracy = (TP + TN) / ...
    (TP + TN + FP + FN);

if (TP + FP) > 0
    Precision = TP / (TP + FP);
else
    Precision = 0;
end

if (TP + FN) > 0
    Recall = TP / (TP + FN);
else
    Recall = 0;
end

if (Precision + Recall) > 0

    F1_Score = ...
        2 * (Precision * Recall) / ...
        (Precision + Recall);

else

    F1_Score = 0;

end

%% -----------------------------------------------------
% DISPLAY RESULTS
% -----------------------------------------------------

fprintf('\n============================================\n');
fprintf('LOGISTIC REGRESSION RESULTS\n');
fprintf('============================================\n');

fprintf('Accuracy  = %.4f\n',Accuracy);
fprintf('Precision = %.4f\n',Precision);
fprintf('Recall    = %.4f\n',Recall);
fprintf('F1 Score  = %.4f\n',F1_Score);

fprintf('\nConfusion Matrix:\n');

fprintf('                 Predicted\n');
fprintf('                 0       1\n');

fprintf('Actual 0        %d      %d\n',TN,FP);
fprintf('Actual 1        %d      %d\n',FN,TP);

%% -----------------------------------------------------
% SAVE RESULTS
% -----------------------------------------------------

LogisticResults = table(...
    Selected_N,...
    Selected_D,...
    Accuracy,...
    Precision,...
    Recall,...
    F1_Score,...
    'VariableNames',{...
    'Cardinality',...
    'Dimension',...
    'Accuracy',...
    'Precision',...
    'Recall',...
    'F1_Score'});

writetable(LogisticResults,...
    fullfile(outputDir,...
    'LogisticRegression_Results.csv'));

save(fullfile(outputDir,...
    'LogisticRegression_Model.mat'),...
    'Logistic_Model',...
    'LogisticResults',...
    'Confusion_Matrix',...
    'Y_test',...
    'Y_pred',...
    'Score');

%% -----------------------------------------------------
% CONFUSION MATRIX FIGURE
% -----------------------------------------------------

figure;

confusionchart(...
    Y_test,...
    Y_pred,...
    'RowSummary','row-normalized',...
    'ColumnSummary','column-normalized');

title(sprintf(...
    'Logistic Regression - N=%d, d=%d',...
    Selected_N,Selected_D));

saveas(gcf,...
    fullfile(outputDir,...
    'LogisticRegression_ConfusionMatrix.png'));

fprintf('\nLogistic Regression completed.\n');

%% =====================================================
% GRAPH 1 - PHASE I AND PHASE II
% =====================================================

figure;

hold on;

for Dim = dimensions

    idx = Results.Dimension == Dim;

    plot( ...
        Results.Cardinality(idx),...
        Results.Time_Phase1(idx),...
        '-o',...
        'DisplayName',...
        sprintf('Phase I - d=%d',Dim));

    plot( ...
        Results.Cardinality(idx),...
        Results.Time_Phase2(idx),...
        '--o',...
        'DisplayName',...
        sprintf('Phase II - d=%d',Dim));

end

xlabel('Cardinality');
ylabel('Execution Time (seconds)');

title('HD Execution Time');

legend('Location','northwest');

grid on;

saveas(gcf,...
    fullfile(outputDir,...
    'HD_Phase_Times.png'));

%% =====================================================
% GRAPH 2 - TOTAL TIME
% =====================================================

figure;

hold on;

for Dim = dimensions

    idx = Results.Dimension == Dim;

    plot( ...
        Results.Cardinality(idx),...
        Results.Time_HD_Total(idx),...
        '-o',...
        'DisplayName',...
        sprintf('Dimension = %d',Dim));

end

xlabel('Cardinality');

ylabel('Total HD Time (seconds)');

title('Effect of Cardinality on HD');

legend('Location','northwest');

grid on;

saveas(gcf,...
    fullfile(outputDir,...
    'HD_Cardinality.png'));

%% =====================================================
% GRAPH 3 - PHASE I
% =====================================================

figure;

hold on;

for Dim = dimensions

    idx = Results.Dimension == Dim;

    plot( ...
        Results.Cardinality(idx),...
        Results.Time_Phase1(idx),...
        '-o',...
        'DisplayName',...
        sprintf('d = %d',Dim));

end

xlabel('Cardinality');
ylabel('Phase I Time (seconds)');

title('Effect of Cardinality on Phase I');

legend('Location','northwest');

grid on;

saveas(gcf,...
    fullfile(outputDir,...
    'Phase1_Cardinality.png'));

%% =====================================================
% GRAPH 4 - PHASE II
% =====================================================

figure;

hold on;

for Dim = dimensions

    idx = Results.Dimension == Dim;

    plot( ...
        Results.Cardinality(idx),...
        Results.Time_Phase2(idx),...
        '-o',...
        'DisplayName',...
        sprintf('d = %d',Dim));

end

xlabel('Cardinality');
ylabel('Phase II Time (seconds)');

title('Effect of Cardinality on Phase II');

legend('Location','northwest');

grid on;

saveas(gcf,...
    fullfile(outputDir,...
    'Phase2_Cardinality.png'));

fprintf('\nAll experiments completed.\n');
fprintf('Results saved in folder: results\n');

%% =====================================================
% GRAPH 5 - EFFECT OF DIMENSION ON TOTAL HD TIME
% =====================================================

figure;

hold on;

for N = cardinalities

    idx = Results.Cardinality == N;

    % Sort by dimension
    TempResults = Results(idx,:);

    TempResults = sortrows(...
        TempResults,...
        'Dimension');

    plot(...
        TempResults.Dimension,...
        TempResults.Time_HD_Total,...
        '-o',...
        'LineWidth',1.2,...
        'MarkerSize',7,...
        'DisplayName',...
        sprintf('N = %d',N));

end

xlabel('Dimension');

ylabel('Total HD Time (seconds)');

title('Effect of Dimension on HD Execution Time');

xticks(dimensions);

legend('Location','northwest');

grid on;

saveas(gcf,...
    fullfile(outputDir,...
    'HD_Dimension.png'));


%% =====================================================
% DATA GENERATION FUNCTION
% =====================================================

function Data = GenerateData(N,d,AM_SUM)

    X = zeros(N,d);
    Y = zeros(N,d);

    %% INPUTS - UNIFORM DISTRIBUTION

    for j = 1:d

        xmin = 10 + 5*AM_SUM + 20*j;

        xmax = xmin + 500 + 150*j;

        X(:,j) = ...
            xmin + ...
            (xmax-xmin).*rand(N,1);

    end

    %% OUTPUTS - NORMAL DISTRIBUTION

    for j = 1:d

        mu = 1000 + ...
             250*AM_SUM + ...
             500*j;

        sigma = 0.12*mu;

        Y(:,j) = ...
            mu + ...
            sigma.*randn(N,1);

        %% Ensure positive outputs

        negative = Y(:,j) <= 0;

        while any(negative)

            Y(negative,j) = ...
                mu + ...
                sigma.*randn(sum(negative),1);

            negative = Y(:,j) <= 0;

        end

    end

    Data = [X Y];

end
