function UrbanInfrastructure_Binary_Class_Roads_Building()
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Problem 2: Extracting urban infrastructure information through satellite images
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% 
% Binary segmentation of satelite images:
%   Class 0 = Road
%   Class 1 = Building
%
% This implementation:
%   - Utilizing the features that emphasize explainable AI
%   - Avoids Image Processing & ML toolboxes
%   - Trains multiple classifiers (KNN, Tree, RF, SVM)
%   - Evaluates performance using accuracy, ROC, AUC, and IoU
%
% Dataset:
%   RGB images with colour-coded pixel annotations
%   Only Road and Building classes are modelled;
%   all other classes are treated as background.
%
% Authors: Veera Suresh Akuthota - A00040066
% Raj -
% 
% ========================================================================================

clc; close all;

%% ----------------- PATHS / SETTINGS -------------------------------------
IMG_DATA  = 'https://github.com/veerasuresh-hub/SateliteImage-Classification/releases/download/v1.0.0/Imgdataset.zip';

% Set dataset paths 
SATIMG  = fullfile(pwd, 'Imgdataset');      
IMG_DIR = fullfile(SATIMG, 'images');
MASK_DIR = fullfile(SATIMG, 'masks');

[IMG_DIR, MASK_DIR] = ensureDataset(IMG_DATA, SATIMG, IMG_DIR, MASK_DIR);

OUT_DIR  = '/MATLAB Drive/out_binary_road_building';
if ~exist(OUT_DIR,'dir'); mkdir(OUT_DIR); end

%[IMG_DIR, MASK_DIR] = ensureDataset(IMG_DATA, SATIMG, IMG_DIR, MSK_DIR);

% Sampling (per class per image)
maxTrainPerClassPerImg = 600;
maxTestPerClassPerImg  = 600;

% Reproducibility
seed = 1;
rng(seed);

%% ----------------- MASK COLOURS (MANUAL) --------------------------------
ROAD_RGB_ANN     = uint8([110 193 228]); % #6EC1E4
BUILDING_RGB_ANN = uint8([ 60  16 152]); % #3C1098
tol = 15; % RGB tolerance for matching (tune 10-30)

%% ----------------- MODELS -----------------------------------------------
predictModelName = 'rf';  % 'svm'|'knn'|'tree'|'rf'
C = 2; % binary

% SVM (one-vs-rest; manual SGD hinge)
svmEpochs = 6;
svmLambda = 1e-4;
svmEta0   = 0.08;

% kNN
kNN_k = 9;

% Tree / RF (manual)
treeMaxDepth      = 5;
treeNumThreshTry  = 30;
treeMaxFeatTry    = 7;
treeMinLeaf       = 80;

rfNumTrees   = 40;
rfSampleFrac = 0.70;

%% ----------------- VIS COLOURS ------------------------------------------
visRoadColor = [0 0 0];        % black
visBldColor  = [1 0 0];        % red
alphaOverlay = 0.40;

%% ----------------- MORPHOLOGY POST-PROC ---------------------------------
doMorph = true;
morphRadius = 1;

doRoadClean = true;
roadLineLen = 17;
minRoadSize = 450;

% stop road flooding by keeping only top-K largest road components
doKeepTopKRoad = true;
keepTopKRoad   = 8;   % try 1..5

% OPTIONAL: keep only elongated road components
doElongatedRoad = true;
elongMinAspect  = 2.5;
elongMinSize    = 200;

% OPTIONAL: keep only roughly rectangular building components
doRectangularBuildings = false;
rectMinFill = 0.45;
rectMinSize = 150;
rectMaxAspect = 4.0;

%% ----------------- FEATURE SETTINGS -------------------------------------
doKMeansFeat = true;
kmeansK = 3;
kmeansIters = 10;
kmeansSampleMax = 6000;

doHoughFeat = true;
houghThetas = [0 45 90 135];
houghTopN   = 6;
houghNearPx = 2;

%% ----------------- DOWNSAMPLING -----------------------------------------
doDownsample = true;
ds = 2 ;

%% ----------------- PCA SETTINGS -----------------------------------------
doPCA = true;
pcaKeepEnergy = 0.95;
useFixedPCAComps = false;
pcaNumComp = 10;

featNames = { ...
    'gray','gradMag','ExG','localVar','edgeDensity', ...
    'R','G','B','meanGray','lapMag','morphGrad','topHat', ...
    'H','S','V','roadColorScore', ...
    'edgeDen0','edgeDen45','edgeDen90','edgeDen135', ...
    'xNorm','yNorm', ...
    'kmeansRoadScore','houghLineScore','lbpMean','lbpVar','hogCorner','linearity'};

doPlotRGBHists   = false;   % RGB channel histograms
doPlotGrayHist   = false;   % grayscale intensity distribution
doPlotClassProps = false;  

%% ----------------- SAVE/LOAD MODEL --------------------------------------
doSaveModel = true;
modelFile   = fullfile(OUT_DIR,'trained_models_binary_road_building.mat');

%% ----------------- EXTERNAL PREDICTION ----------------------------------
PREDICT_IMAGE_LIST = {
   '/MATLAB Drive/images/image0008.jpg'
   '/MATLAB Drive/images/image0011.jpg'
};

%% ----------------- LIST FILES / SPLIT -----------------------------------
imgFiles = dir(fullfile(IMG_DIR,'image*.jpg'));
nImages = numel(imgFiles);
if nImages == 0
    error('No images found in %s', IMG_DIR);
end

perm = randperm(nImages);
nTrain = floor(0.8*nImages);
trainIdx = perm(1:nTrain);
testIdx  = perm(nTrain+1:end);

fprintf('Images: %d | Train: %d | Test: %d\n', nImages, numel(trainIdx), numel(testIdx));
if doDownsample
    fprintf('Downsampling enabled: ds=%d (stride)\n', ds);
else
    fprintf('Downsampling disabled (full resolution)\n');
end
if doPCA
    fprintf('PCA enabled: keepEnergy=%.2f (or fixed=%d comps=%d)\n', pcaKeepEnergy, useFixedPCAComps, pcaNumComp);
end

%% ----------------- BUILD TRAIN SET (BINARY) -----------------------------
% Storage for training feature vectors and labels
Xtr = [];
ytr = [];   % 0 = road, 1 = building

% Loop over training images to collect labelled pixels
for ii = trainIdx
    imgName = imgFiles(ii).name;
    imgPath = fullfile(IMG_DIR, imgName);
    maskPath = deriveMaskPath(imgName, MASK_DIR);
    if isempty(maskPath) || ~exist(maskPath,'file')
        warning('Mask missing for %s. Skipping.', imgName);
        continue;
    end

    I0 = imread(imgPath);
    M0 = imread(maskPath);

    if doDownsample && ds > 1
        I = I0(1:ds:end, 1:ds:end, :);
        M = M0(1:ds:end, 1:ds:end, :);
    else
        I = I0; M = M0;
    end

   % Decode colour-coded annotation mask into binary road/building masks
   [isRoad, isBld] = decodeRoadBuildingMasks(M, ROAD_RGB_ANN, BUILDING_RGB_ANN, tol);

    roadIdxAll = find(isRoad);
    bldIdxAll  = find(isBld);

    if isempty(roadIdxAll) && isempty(bldIdxAll)
        warning('Missing road/building pixels in %s. Skipping.', imgName);
        continue;
    end

    feats = computeFeatures_noBuiltins_withMorph( ...
        I, morphRadius, ...
        doKMeansFeat, kmeansK, kmeansIters, kmeansSampleMax, ...
        doHoughFeat, houghThetas, houghTopN, houghNearPx);

    nR = numel(roadIdxAll);
    nB = numel(bldIdxAll);

    nRS = min(nR, maxTrainPerClassPerImg);
    nBS = min(nB, maxTrainPerClassPerImg);

    roadPick = roadIdxAll(randperm(nR, nRS));
    bldPick  = bldIdxAll(randperm(nB, nBS));

    idx = [roadPick; bldPick];
    lab = [zeros(numel(roadPick),1); ones(numel(bldPick),1)]; % 0=Road, 1=Building

    Xtr = [Xtr; feats(idx,:)];
    ytr = [ytr; lab];
end

if isempty(Xtr)
    error('No training pixels collected.');
end

% -------- Global balancing (BINARY) --------
n0 = sum(ytr==0);
n1 = sum(ytr==1);
if n0==0 || n1==0
    error('Missing class after sampling. road=%d building=%d', n0, n1);
end

m = min(n0,n1);
mCap = 25000;
m = min(m, mCap);

i0 = find(ytr==0); i1 = find(ytr==1);
i0 = i0(randperm(n0, m));
i1 = i1(randperm(n1, m));

sel = [i0; i1];
sel = sel(randperm(numel(sel)));

Xtr = Xtr(sel,:);
ytr = ytr(sel);

fprintf('Train pixels (balanced): %d  [road=%d building=%d]\n', numel(ytr), sum(ytr==0), sum(ytr==1));

% ================== PLOT: RGB HISTOGRAMS (TRAIN) ==================
% Feature order in Xtr is: ... R(6), G(7), B(8) ...
if doPlotRGBHists
plotRGBHistograms_binary(Xtr, ytr, 60);
drawnow;
end

%% -------- PIXEL INTENSITY DISTRIBUTION -----------------------
% Visualise the distribution of grayscale pixel intensities for the
% training data to analyse variability and detect potential imbalance.

if doPlotGrayHist
figure('Name','Grayscale Pixel Intensity Distribution','NumberTitle','off');
histogram(Xtr(:,1), 50, 'Normalization','pdf');   % Feature 1 = gray
xlabel('Grayscale Intensity');
ylabel('Probability Density');
title('Grayscale Pixel Intensity Distribution (Training Data)');
grid on;
drawnow;
% Save figure
saveas(gcf, fullfile(OUT_DIR,'gray_intensity_distribution.png'));
end



%% ----------------- STANDARDISE ------------------------------------------
[mu, sd] = meanStd_manual(Xtr);
XtrZ = zscore_manual(Xtr, mu, sd);


%% ================= DESCRIPTIVE FEATURE STATISTICS ======================
fprintf('\n=== DESCRIPTIVE STATISTICS: TRAINING FEATURES ===\n');

featStats.mean     = mean(Xtr,1);
featStats.median   = median(Xtr,1);
featStats.std      = std(Xtr,0,1);
featStats.skewness = skewness(Xtr,0,1);
featStats.kurtosis = kurtosis(Xtr,0,1);

for f = 1:min(numel(featNames), size(Xtr,2))
    fprintf('%-22s | mean=%8.4f  std=%8.4f  skew=%7.3f  kurt=%7.3f\n', ...
        featNames{f}, ...
        featStats.mean(f), ...
        featStats.std(f), ...
        featStats.skewness(f), ...
        featStats.kurtosis(f));
end

%% ------------------ PCA DIMENSIONALITY REDUCTION ------------------------
% Principal Component Analysis is applied after feature normalisation
% to reduce redundancy and improve classifier stability.
% Components are retained to preserve 95% of total variance.

P = []; kPCA = size(XtrZ,2); cumExplained = [];
if doPCA
    if useFixedPCAComps
        [P, kPCA, cumExplained] = pca_fit_manual_eig(XtrZ, 0, true, pcaNumComp);
    else
        [P, kPCA, cumExplained] = pca_fit_manual_eig(XtrZ, pcaKeepEnergy, false, 0);
    end
    XtrZ = XtrZ * P(:,1:kPCA);
    if ~isempty(cumExplained)
        fprintf('[PCA] Reduced D=%d -> k=%d | retained variance = %.2f%%\n', size(P,1), kPCA, 100*cumExplained(kPCA));
    end
end

%% ================= CLASS-WISE FEATURE STATISTICS =======================
fprintf('\n=== CLASS-WISE FEATURE COMPARISON (ROAD vs BUILDING) ===\n');

roadIdx = (ytr == 0);
bldIdx  = (ytr == 1);

for f = 1:min(numel(featNames), size(Xtr,2))
    muR = mean(Xtr(roadIdx,f));
    muB = mean(Xtr(bldIdx,f));
    fprintf('%-22s | RoadMean=%7.4f  BldMean=%7.4f  Diff=%7.4f\n', ...
        featNames{f}, muR, muB, muB-muR);
end



%% ----------------- MODEL TRAINING --------------------------------
fprintf('\nTraining models for BINARY road/building...\n');
fprintf('KNN uses training set directly (k=%d).\n', kNN_k);

fprintf('Training Decision Tree (binary)...\n');
tree = trainTree_multiclass_manual(XtrZ, ytr, C, treeMaxDepth, treeNumThreshTry, treeMaxFeatTry, treeMinLeaf);

fprintf('Training Random Forest (binary)...\n');
rf = trainRF_multiclass_manual(XtrZ, ytr, C, rfNumTrees, rfSampleFrac, treeMaxDepth, treeNumThreshTry, treeMaxFeatTry, treeMinLeaf);

fprintf('Training Linear SVM OVR (SGD) (binary)...\n');
[Wovr, bovr] = trainSVM_OVR_manual(XtrZ, ytr, C, svmEpochs, svmLambda, svmEta0);

%% ----------------- SAVE TRAINED MODELS ----------------------------------
if doSaveModel
    save(modelFile, ...
        'XtrZ','ytr','kNN_k', ...
        'mu','sd','doDownsample','ds', ...
        'doMorph','morphRadius','doRoadClean','roadLineLen','minRoadSize', ...
        'doKeepTopKRoad','keepTopKRoad', ...
        'doElongatedRoad','elongMinAspect','elongMinSize', ...
        'doRectangularBuildings','rectMinFill','rectMinSize','rectMaxAspect', ...
        'doPCA','P','kPCA','cumExplained','pcaKeepEnergy','useFixedPCAComps','pcaNumComp', ...
        'tree','rf', ...
        'Wovr','bovr', ...
        'ROAD_RGB_ANN','BUILDING_RGB_ANN','tol', ...
        'visRoadColor','visBldColor','alphaOverlay', ...
        'doKMeansFeat','kmeansK','kmeansIters','kmeansSampleMax', ...
        'doHoughFeat','houghThetas','houghTopN','houghNearPx', ...
        'featNames');

    fprintf('\n[MODEL] Saved trained model to:\n  %s\n', modelFile);
end

%% ----------------- BUILD TEST SET (BINARY) ------------------------------
Xte = [];
yte = [];

imgCount = 0;
testImageN   = struct('name',{},'n',{});

for ii = testIdx
    imgName = imgFiles(ii).name;
    imgPath = fullfile(IMG_DIR, imgName);
    maskPath = deriveMaskPath(imgName, MASK_DIR);
    if isempty(maskPath) || ~exist(maskPath,'file')
        continue;
    end

    I0 = imread(imgPath);
    M0 = imread(maskPath);

    if doDownsample && ds > 1
        I = I0(1:ds:end, 1:ds:end, :);
        M = M0(1:ds:end, 1:ds:end, :);
    else
        I = I0; M = M0;
    end

    [isRoad, isBld] = decodeRoadBuildingMasks(M, ROAD_RGB_ANN, BUILDING_RGB_ANN, tol);
    roadIdxAll = find(isRoad);
    bldIdxAll  = find(isBld);

    if isempty(roadIdxAll) || isempty(bldIdxAll)
        continue;
    end

    feats = computeFeatures_noBuiltins_withMorph( ...
        I, morphRadius, ...
        doKMeansFeat, kmeansK, kmeansIters, kmeansSampleMax, ...
        doHoughFeat, houghThetas, houghTopN, houghNearPx);

    nRS = min(numel(roadIdxAll), maxTestPerClassPerImg);
    nBS = min(numel(bldIdxAll),  maxTestPerClassPerImg);

    roadPick = roadIdxAll(randperm(numel(roadIdxAll), nRS));
    bldPick  = bldIdxAll(randperm(numel(bldIdxAll),  nBS));

    idx = [roadPick; bldPick];
    lab = [zeros(numel(roadPick),1); ones(numel(bldPick),1)];

    Xte = [Xte; feats(idx,:)];
    yte = [yte; lab];

    imgCount = imgCount + 1;
    testImageN(imgCount).name = imgName;
    testImageN(imgCount).n    = numel(lab);
end

if isempty(Xte)
    warning('No test pixels collected (evaluation skipped).');
else
    % ---- SAFETY CHECK BEFORE ZSCORE ----
fprintf('DEBUG: size(Xtr) = %dx%d\n', size(Xtr,1), size(Xtr,2));
fprintf('DEBUG: size(Xte) = %dx%d\n', size(Xte,1), size(Xte,2));
fprintf('DEBUG: numel(mu) = %d | numel(sd) = %d\n', numel(mu), numel(sd));

if size(Xte,2) ~= numel(mu) || size(Xte,2) ~= numel(sd)
    error('Feature dimension mismatch: Xte has %d cols, but mu/sd have %d/%d. Check computeFeatures output is same for train and test.', ...
        size(Xte,2), numel(mu), numel(sd));
end
        XteZ = zscore_manual(Xte, mu, sd);
    if doPCA
        XteZ = XteZ * P(:,1:kPCA);
    end

    fprintf('\nTest pixels: %d  [road=%d building=%d]\n', numel(yte), sum(yte==0), sum(yte==1));
    fprintf('\n=== RESULTS (BINARY: 0=Road, 1=Building) ===\n\n');

%% ================= INFERENTIAL STATS (T-TEST) ==========================
grayIdx = strcmp(featNames,'gray');

grayRoad = Xtr(ytr==0, grayIdx);
grayBld  = Xtr(ytr==1, grayIdx);

[~,pVal,~,stats] = ttest2(grayRoad, grayBld);

fprintf('Gray feature t-test:\n');
fprintf('t = %.4f | p-value = %.6f\n', stats.tstat, pVal);

if pVal < 0.05
    fprintf('=> Statistically significant difference (p < 0.05)\n\n');
else
    fprintf('=> No statistically significant difference\n\n');
end


    % --- KNN ---
    predKNN = knnPredict_multiclass_manual(XtrZ, ytr, XteZ, kNN_k, C);
    scoreKNN = knnScore_binary_manual(XtrZ, ytr, XteZ, kNN_k); % P(building)
    printBinaryReportPlusROC('KNN', yte, predKNN, scoreKNN);

    % --- Tree ---
    [predTree, probTree] = predictTree_multiclass_manual(tree, XteZ);
    scoreTree = probTree(:,2);
    printBinaryReportPlusROC('Tree', yte, predTree, scoreTree);

    % --- RF ---
    [predRF, probRF] = predictRF_multiclass_manual(rf, XteZ);
    scoreRF = probRF(:,2);
    printBinaryReportPlusROC('RF', yte, predRF, scoreRF);

  
    %% ================= CONFIDENCE INTERVAL (RF ACCURACY) ===================
accRF = mean(predRF == yte);
n = numel(yte);
z = 1.96; % 95% CI

ciLow  = accRF - z*sqrt((accRF*(1-accRF))/n);
ciHigh = accRF + z*sqrt((accRF*(1-accRF))/n);

fprintf('RF Accuracy = %.4f\n', accRF);
fprintf('95%% Confidence Interval = [%.4f , %.4f]\n\n', ciLow, ciHigh);

    % --- SVM ---
    predSVM = svmPredict_OVR_manual(XteZ, Wovr, bovr);
    scoreSVM = svmScore_binary_manual(XteZ, Wovr, bovr); % raw score for building
    printBinaryReportPlusROC('SVM(OVR)', yte, predSVM, scoreSVM);

    % ===== PLOT: Road vs Building proportions (GT + models) =====
    
    if doPlotClassProps
    plotClassProportions_binary(yte, ...
    {predSVM, predKNN, predTree, predRF}, ...
    {'SVM','KNN','Tree','RF'});
    drawnow;
    end

    % ================== DASHBOARD (6 PLOTS) ==================
    showPR = true;   % or false
    plotDashboard6_binary( ...
    Xtr, ytr, ...
    yte, ...
    predKNN, predTree, predRF, predSVM, ...
    scoreKNN, scoreTree, scoreRF, scoreSVM, ...
    featNames, ...
    showPR);

%% ------------------ INTERSECTION OVER UNION (IoU) -----------------------
% IoU is computed only on pixels belonging to Road or Building.
% Background pixels are excluded to avoid bias from dominant background.
% Due to computational cost, evaluation is limited to few images.    

fprintf('\n=== IoU (EXCLUDING BACKGROUND, FULL IMAGE) [RF ONLY - FAST] ===\n');

    R = evalIoU_onTestImages_binary_FAST( ...
        imgFiles, testIdx, IMG_DIR, MASK_DIR, ...
        ROAD_RGB_ANN, BUILDING_RGB_ANN, tol, ...
        doDownsample, ds, ...
        mu, sd, doPCA, P, kPCA, ...
        rf, ...
        morphRadius, ...
        doKMeansFeat, kmeansK, kmeansIters, kmeansSampleMax, ...
        doHoughFeat, houghThetas, houghTopN, houghNearPx);

    fprintf('RF        IoU-Road=%.4f  IoU-Building=%.4f  mIoU=%.4f\n', ...
        R.meanIoURoad, R.meanIoUBld, R.meanmIoU);
    fprintf('\n');

    % ---------- ACCURACY DESCRIPTIVE STATS (overall) ----------
   fprintf('\n=== ACCURACY DESCRIPTIVE STATS (Overall Test Set) ===\n');
    accAll = [ ...
      mean(predKNN==yte), ...
      mean(predTree==yte), ...
      mean(predRF==yte), ...
      mean(predSVM==yte) ];
names = {'KNN','Tree','RF','SVM'};
for i=1:numel(names)
    fprintf('%-9s Acc=%.4f\n', names{i}, accAll(i));
end

fprintf('Mean Acc across models = %.4f | Std = %.4f | Min = %.4f | Max = %.4f\n', ...
    mean(accAll), std(accAll), min(accAll), max(accAll));

   
end

%% ----------------- INFERENCE ON NEW IMAGES ------------------------------
if ~isempty(PREDICT_IMAGE_LIST)
    fprintf('\n[INFERENCE] Predicting on NEW images...\n');
    for i=1:numel(PREDICT_IMAGE_LIST)
        inImg = PREDICT_IMAGE_LIST{i};
        if ~exist(inImg,'file')
            warning('Missing image: %s', inImg);
            continue;
        end
        [~,base,~] = fileparts(inImg);

        % OPTIONAL: if annotation mask exists for this image, export BG mask
        maskPath = deriveMaskPath([base '.jpg'], MASK_DIR); % assumes base matches image#### pattern
        if ~isempty(maskPath) && exist(maskPath,'file')
            M0 = imread(maskPath);
            bgFull = backgroundMaskFromAnnotation(M0, ROAD_RGB_ANN, BUILDING_RGB_ANN, tol);
            outBgFull = fullfile(OUT_DIR, ['gt_' base '_BACKGROUND_FULL.png']);
            imwrite(bgFull, outBgFull);
            fprintf('    GT background FULL: %s\n', outBgFull);

            if doDownsample && ds>1
                bgSmall = bgFull(1:ds:end, 1:ds:end);
            else
                bgSmall = bgFull;
            end
            outBgSmall = fullfile(OUT_DIR, ['gt_' base '_BACKGROUND_SMALL.png']);
            imwrite(bgSmall, outBgSmall);
            fprintf('    GT background SMALL: %s\n', outBgSmall);
        end

        outLblSmall = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_label_SMALL.png']);
        outLblFull  = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_label_FULL.png']);

        outBldSmall = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_BUILDING_SMALL.png']);
        outBldFull  = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_BUILDING_FULL.png']);

        outRoadSmall = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_ROAD_SMALL.png']);
        outRoadFull  = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_ROAD_FULL.png']);

        outColSmall = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_color_SMALL.png']);
        outColFull  = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_color_FULL.png']);

        outOvSmall  = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_overlay_SMALL.png']);
        outOvFull   = fullfile(OUT_DIR, ['pred_' base '_' upper(predictModelName) '_overlay_FULL.png']);

        predict_from_saved_model_binary( ...
            modelFile, inImg, predictModelName, ...
            outLblSmall, outLblFull, outBldSmall, outBldFull, outRoadSmall, outRoadFull, ...
            outColSmall, outColFull, outOvSmall, outOvFull);
    end
    fprintf('[INFERENCE] Done.\n');
else
    fprintf('\n[INFERENCE] PREDICT_IMAGE_LIST empty -> skipping.\n');
end

end % ===================== END MAIN =======================================

%% ============================ USER DEFINED FUNCTIONS ===================================

function maskPath = deriveMaskPath(imgName, maskDir)
tok = regexp(imgName, '^image(\d+)\.jpg$', 'tokens', 'once');
if isempty(tok)
    maskPath = '';
else
    maskPath = fullfile(maskDir, ['mask' tok{1} '.png']);
end
end

function bgMask255 = backgroundMaskFromAnnotation(maskImg, roadRGB, bldRGB, tol)
% Returns uint8 mask: 255 = background (NOT road/building), 0 = road/building
[isRoad, isBld] = decodeRoadBuildingMasks(maskImg, roadRGB, bldRGB, tol);
isBG = ~(isRoad | isBld);
bgMask255 = uint8(isBG) * 255;
end

function [isRoad, isBld] = decodeRoadBuildingMasks(M, roadRGB, bldRGB, tol)
% DECODEROADBUILDINGMASKS
% =========================================================================
% Converts colour-coded annotation masks into binary logical masks.
%
% Inputs:
%   M        - RGB annotation mask image
%   roadRGB - RGB triplet for road class
%   bldRGB  - RGB triplet for building class
%   tol     - Colour tolerance for matching
%
% Outputs:
%   isRoad  - Logical mask of road pixels
%   isBld   - Logical mask of building pixels
%
% Notes:
%   - All other colours are treated as background
%   - Building pixels take precedence if overlap occurs
% =========================================================================
R = int16(M(:,:,1)); G = int16(M(:,:,2)); B = int16(M(:,:,3));
rd = int16(roadRGB(:)'); bd = int16(bldRGB(:)');

isRoad = (abs(R-rd(1))<=tol) & (abs(G-rd(2))<=tol) & (abs(B-rd(3))<=tol);
isBld  = (abs(R-bd(1))<=tol) & (abs(G-bd(2))<=tol) & (abs(B-bd(3))<=tol);

% priority: building > road (in case tolerance overlaps)
isRoad = isRoad & ~isBld;
end

function [iouRoad, iouBld] = iouBinary_excludingBackground(gtRoad, gtBld, predLbl)
valid = gtRoad | gtBld;

gtLbl = zeros(size(predLbl), 'uint8');
gtLbl(gtBld) = 1;

p = predLbl(valid);
g = gtLbl(valid);

TP0 = sum((p==0) & (g==0));
FP0 = sum((p==0) & (g==1));
FN0 = sum((p==1) & (g==0));
iouRoad = TP0 / (TP0 + FP0 + FN0 + eps);

TP1 = sum((p==1) & (g==1));
FP1 = sum((p==1) & (g==0));
FN1 = sum((p==0) & (g==1));
iouBld = TP1 / (TP1 + FP1 + FN1 + eps);
end

function results = evalIoU_onTestImages_binary_FAST( ...
    imgFiles, testIdx, IMG_DIR, MASK_DIR, ...
    ROAD_RGB_ANN, BUILDING_RGB_ANN, tol, ...
    doDownsample, ds, ...
    mu, sd, doPCA, P, kPCA, ...
    rf, ...
    morphRadius, ...
    doKMeansFeat, kmeansK, kmeansIters, kmeansSampleMax, ...
    doHoughFeat, houghThetas, houghTopN, houghNearPx)

sumIoUR = 0;
sumIoUB = 0;
nUsed   = 0;

MAX_IOU_IMAGES = 1;   % Images to process
for kk = 1:min(MAX_IOU_IMAGES, numel(testIdx))
    ii = testIdx(kk);

    imgName = imgFiles(ii).name;
    imgPath = fullfile(IMG_DIR, imgName);
    maskPath = deriveMaskPath(imgName, MASK_DIR);
    if isempty(maskPath) || ~exist(maskPath,'file')
        continue;
    end

    fprintf('  [IoU RF] %d/%d  %s\n', kk, numel(testIdx), imgName);

    I0 = imread(imgPath);
    M0 = imread(maskPath);

    if doDownsample && ds > 1
        I = I0(1:ds:end, 1:ds:end, :);
        M = M0(1:ds:end, 1:ds:end, :);
    else
        I = I0;
        M = M0;
    end

    [gtRoad, gtBld] = decodeRoadBuildingMasks(M, ROAD_RGB_ANN, BUILDING_RGB_ANN, tol);
    valid = gtRoad | gtBld;
    if ~any(valid(:))
        continue;
    end

    F  = computeFeatures_noBuiltins_withMorph( ...
        I, morphRadius, ...
        doKMeansFeat, kmeansK, kmeansIters, kmeansSampleMax, ...
        doHoughFeat, houghThetas, houghTopN, houghNearPx);

    Fz = zscore_manual(F, mu, sd);
    if doPCA
        Fz = Fz * P(:,1:kPCA);
    end

    H = size(I,1);
    W = size(I,2);

    [pred, ~] = predictRF_multiclass_manual(rf, Fz);
    predLbl = uint8(reshape(pred, H, W));

    [iouR, iouB] = iouBinary_excludingBackground(gtRoad, gtBld, predLbl);

    sumIoUR = sumIoUR + iouR;
    sumIoUB = sumIoUB + iouB;
    nUsed   = nUsed + 1;
end

results = struct();
if nUsed == 0
    results.meanIoURoad = 0;
    results.meanIoUBld  = 0;
    results.meanmIoU    = 0;
else
    results.meanIoURoad = sumIoUR / nUsed;
    results.meanIoUBld  = sumIoUB / nUsed;
    results.meanmIoU    = 0.5*(results.meanIoURoad + results.meanIoUBld);
end
end

%% ----------------- FEATURES EXTRACTION---------------------------------------------
% FEATURE EXTRACTION FUNCTION
% -------------------------------------------------------------------------
% This function computes a set of essential features for each pixel:
%   - Intensity and colour features
%   - Gradient and edge features
%   - Texture descriptors (LBP, HOG, structure tensor)
%   - Geometric and spatial priors
%   - Optional K-means and Hough-based features
% -------------------------------------------------------------------------
function F = computeFeatures_noBuiltins_withMorph( ...
    rgbImg, morphRadius, ...
    doKMeansFeat, kmeansK, kmeansIters, kmeansSampleMax, ...
    doHoughFeat, houghThetas, houghTopN, houghNearPx)

% === 1. Colour and intensity features ===
I = double(rgbImg) / 255.0;
R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
[H,W] = size(R);

% === 2. Gradient and edge-based features ===
gray = 0.2989*R + 0.5870*G + 0.1140*B;

sx = [-1 0 1; -2 0 2; -1 0 1];
sy = [ 1 2 1;  0 0 0; -1 -2 -1];
Gx = conv2_manual(gray, sx);
Gy = conv2_manual(gray, sy);
gradMag = sqrt(Gx.^2 + Gy.^2);

exgRaw = 2*G - R - B;
exgMin = min(exgRaw(:));
exgMax = max(exgRaw(:));
if exgMax > exgMin
    exg = (exgRaw - exgMin) / (exgMax - exgMin);
else
    exg = zeros(H,W);
end

k7 = ones(7,7) / 49;
meanGray = conv2_manual(gray, k7);
meanGraySq = conv2_manual(gray.^2, k7);
localVar = meanGraySq - meanGray.^2;
localVar(localVar < 0) = 0;

th = 0.1 * max(gradMag(:));
edgeMap = double(gradMag > th);
k9 = ones(9,9) / 81;
edgeDensity = conv2_manual(edgeMap, k9);

lapK = [0 -1 0; -1 4 -1; 0 -1 0];
lap = conv2_manual(gray, lapK);
lapMag = abs(lap);

% === 3. Morphological features ===
se = makeDiskSE_manual(morphRadius);
dil = grayDilate_manual(gray, se);
ero = grayErode_manual(gray, se);
morphGrad = dil - ero;

opened = grayDilate_manual(grayErode_manual(gray, se), se);
topHat = gray - opened;

[Hh, Ss, Vv] = rgb2hsv_manual(I);

sigmaV = 0.18;
lowSat = 1 - Ss;
midVal = exp(-((Vv - 0.55).^2) / (2*sigmaV*sigmaV));
roadColorScore = lowSat .* midVal;

ori = atan2(Gy, Gx) * (180/pi);
ori(ori < 0) = ori(ori < 0) + 180;

edgeStrong = (gradMag > th);
bw0   = edgeStrong & ( (ori <= 22.5) | (ori >= 157.5) );
bw45  = edgeStrong & ( ori > 22.5  & ori <= 67.5 );
bw90  = edgeStrong & ( ori > 67.5  & ori <= 112.5 );
bw135 = edgeStrong & ( ori > 112.5 & ori < 157.5 );

edgeDen0   = conv2_manual(double(bw0),   k9);
edgeDen45  = conv2_manual(double(bw45),  k9);
edgeDen90  = conv2_manual(double(bw90),  k9);
edgeDen135 = conv2_manual(double(bw135), k9);

xNorm = zeros(H,W);
yNorm = zeros(H,W);
for r=1:H
    yn = (r-1) / max(1,(H-1));
    for c=1:W
        xNorm(r,c) = (c-1) / max(1,(W-1));
        yNorm(r,c) = yn;
    end
end

[lbpMean, lbpVar] = lbp8_stats_manual(gray, 7);
hogCorner = harrisCornerScore_manual(Gx, Gy, 5, 0.04);
linearity = structureTensorLinearity_manual(Gx, Gy, 7);

kmeansRoadScore = zeros(H,W);
if doKMeansFeat
    Xkm = [Hh(:), Ss(:), Vv(:), gray(:)];
    [kmC, ~] = kmeans_manual_fit_predict(Xkm, kmeansK, kmeansIters, kmeansSampleMax);
    roadCluster = pickRoadCluster_manual(kmC);
    d = sqDistToCentroid_manual(Xkm, kmC(roadCluster,:));
    d = reshape(d, H, W);
    dMin = min(d(:)); dMax = max(d(:));
    if dMax > dMin
        kmeansRoadScore = 1 - (d - dMin) / (dMax - dMin);
    end
end

houghLineScore = zeros(H,W);
if doHoughFeat
    E = (gradMag > th);
    lines = hough_lines_manual(E, houghThetas, houghTopN);
    houghLineScore = lineProximityScore_manual(H, W, lines, houghNearPx);
end

F = [gray(:), gradMag(:), exg(:), localVar(:), edgeDensity(:), ...
     R(:), G(:), B(:), meanGray(:), lapMag(:), morphGrad(:), topHat(:), ...
     Hh(:), Ss(:), Vv(:), roadColorScore(:), ...
     edgeDen0(:), edgeDen45(:), edgeDen90(:), edgeDen135(:), ...
     xNorm(:), yNorm(:), ...
     kmeansRoadScore(:), houghLineScore(:), lbpMean(:), lbpVar(:), hogCorner(:), linearity(:)];
end

function [H,S,V] = rgb2hsv_manual(I)
R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
Cmax = max(max(R,G),B);
Cmin = min(min(R,G),B);
d = Cmax - Cmin;
V = Cmax;
S = zeros(size(Cmax));
nz = (Cmax > 1e-12);
S(nz) = d(nz) ./ Cmax(nz);
H = zeros(size(Cmax));
m = (d > 1e-12);
idx = m & (Cmax == R);
H(idx) = mod((G(idx) - B(idx)) ./ d(idx), 6) / 6;
idx = m & (Cmax == G);
H(idx) = ((B(idx) - R(idx)) ./ d(idx) + 2) / 6;
idx = m & (Cmax == B);
H(idx) = ((R(idx) - G(idx)) ./ d(idx) + 4) / 6;
H(H < 0) = H(H < 0) + 1;
H(H >= 1) = H(H >= 1) - 1;
end

function out = conv2_manual(img, kernel)
[H,W] = size(img);
[kH,kW] = size(kernel);
pH = floor(kH/2);
pW = floor(kW/2);
out = zeros(H,W);
k = rot90(kernel,2);
for i = 1:H
    for j = 1:W
        s = 0.0;
        for a = 1:kH
            ii = i + (a - (pH+1));
            if ii < 1 || ii > H, continue; end
            for b = 1:kW
                jj = j + (b - (pW+1));
                if jj < 1 || jj > W, continue; end
                s = s + img(ii,jj) * k(a,b);
            end
        end
        out(i,j) = s;
    end
end
end

%% ====================== FEATURE NORMALISATION & PCA =====================
function [mu, sd] = meanStd_manual(X)
[N,D] = size(X);
mu = zeros(1,D);
sd = zeros(1,D);
for d=1:D
    s = 0.0;
    for i=1:N, s = s + X(i,d); end
    mu(d) = s / N;
    v = 0.0;
    for i=1:N
        diff = X(i,d) - mu(d);
        v = v + diff*diff;
    end
    sd(d) = sqrt(v / max(1,(N-1)));
    if sd(d) < 1e-12, sd(d) = 1e-12; end
end
end

function Z = zscore_manual(X, mu, sd)
[N,D] = size(X);
Z = zeros(N,D);
for i=1:N
    for d=1:D
        Z(i,d) = (X(i,d) - mu(d)) / sd(d);
    end
end
end

function [P, k, cumExplained] = pca_fit_manual_eig(Xz, keepEnergy, fixedK, kFixed)
[N,D] = size(Xz);
C = (Xz' * Xz) / max(1,(N-1));
[V, L] = eig(C);
eigvals = diag(L);
[~, idx] = sort(eigvals, 'descend');
P = V(:, idx);
eigvalsSorted = eigvals(idx);
total = sum(eigvalsSorted);
if total > 0
    cumExplained = cumsum(eigvalsSorted) / total;
else
    cumExplained = ones(size(eigvalsSorted));
end
if fixedK
    k = min(D, max(1, kFixed));
else
    k = find(cumExplained >= keepEnergy, 1, 'first');
    if isempty(k), k = D; end
end
end

%% ----------------- KNN MULTICLASS ---------------------------------------
function pred = knnPredict_multiclass_manual(Xtrain, yTrain, Xtest, k, C)
nTest  = size(Xtest,1);
nTrain = size(Xtrain,1);
k = min(k, nTrain);
if k < 1
    error('KNN: no training samples available.');
end
pred = zeros(nTest,1);
batch = 80;
for s = 1:batch:nTest
    e = min(nTest, s+batch-1);
    for i = s:e
        dists = zeros(nTrain,1);
        for j = 1:nTrain
            d = 0.0;
            for f = 1:size(Xtrain,2)
                diff = Xtest(i,f) - Xtrain(j,f);
                d = d + diff*diff;
            end
            dists(j) = d;
        end
        idxK = kSmallestIndices(dists, k);
        votes = yTrain(idxK);
        cnt = zeros(C,1);
        for t=1:numel(votes)
            cls = votes(t)+1;
            cnt(cls) = cnt(cls) + 1;
        end
        [~,mx] = max(cnt);
        pred(i) = mx-1;
    end
end
end

function score = knnScore_binary_manual(Xtrain, yTrain, Xtest, k)
nTest  = size(Xtest,1);
nTrain = size(Xtrain,1);
k = min(k, nTrain);
score = zeros(nTest,1);
batch = 80;
for s = 1:batch:nTest
    e = min(nTest, s+batch-1);
    for i = s:e
        dists = zeros(nTrain,1);
        for j = 1:nTrain
            d = 0.0;
            for f = 1:size(Xtrain,2)
                diff = Xtest(i,f) - Xtrain(j,f);
                d = d + diff*diff;
            end
            dists(j) = d;
        end
        idxK = kSmallestIndices(dists, k);
        votes = yTrain(idxK);
        score(i) = mean(votes==1);
    end
end
end

function idx = kSmallestIndices(v, k)
n = numel(v);
idx = zeros(k,1);
tmp = v;
for t=1:k
    best = 1; bestVal = tmp(1);
    for i=2:n
        if tmp(i) < bestVal
            bestVal = tmp(i);
            best = i;
        end
    end
    idx(t) = best;
    tmp(best) = inf;
end
end

%% ----------------- TREE/RF MULTICLASS -----------------------------------
function tree = trainTree_multiclass_manual(X, y, C, maxDepth, nThreshTry, nFeatTry, minLeaf)
tree = buildNode_mc(X, y, C, 0, maxDepth, nThreshTry, nFeatTry, minLeaf);
end

function node = buildNode_mc(X, y, C, depth, maxDepth, nThreshTry, nFeatTry, minLeaf)
node = struct();
N = size(X,1);
p = zeros(1,C);
for c=0:C-1
    p(c+1) = sum(y==c) / max(1,N);
end
node.p = p;

if depth >= maxDepth || N <= minLeaf || all(y==y(1))
    node.isLeaf = true;
    node.pred = argmax01(p)-1;
    return;
end

D = size(X,2);
featIdx = randperm(D, min(nFeatTry, D));

bestImp = inf;
bestF = 1;
bestT = 0;

rowPick = randperm(N, min(N, nThreshTry));
for fi = 1:numel(featIdx)
    f = featIdx(fi);
    for rp = 1:numel(rowPick)
        t = X(rowPick(rp), f);
        left = X(:,f) <= t;
        nL = sum(left);
        nR = N - nL;
        if nL < 10 || nR < 10, continue; end
        yL = y(left);
        yR = y(~left);
        g = (nL/N)*gini_mc(yL,C) + (nR/N)*gini_mc(yR,C);
        if g < bestImp
            bestImp = g;
            bestF = f;
            bestT = t;
        end
    end
end

if ~isfinite(bestImp)
    node.isLeaf = true;
    node.pred = argmax01(p)-1;
    return;
end

node.isLeaf = false;
node.f = bestF;
node.t = bestT;

left = X(:,bestF) <= bestT;
node.left  = buildNode_mc(X(left,:),  y(left),  C, depth+1, maxDepth, nThreshTry, nFeatTry, minLeaf);
node.right = buildNode_mc(X(~left,:), y(~left), C, depth+1, maxDepth, nThreshTry, nFeatTry, minLeaf);
end

function g = gini_mc(y, C)
n = numel(y);
if n==0, g=0; return; end
s = 0.0;
for c=0:C-1
    p = sum(y==c)/n;
    s = s + p*p;
end
g = 1 - s;
end

function k = argmax01(v)
best = 1; bestVal = v(1);
for i=2:numel(v)
    if v(i) > bestVal
        bestVal = v(i);
        best = i;
    end
end
k = best;
end

function [pred, prob] = predictTree_multiclass_manual(tree, X)
n = size(X,1);
C = numel(tree.p);
pred = zeros(n,1);
prob = zeros(n,C);
for i=1:n
    node = tree;
    while ~node.isLeaf
        if X(i,node.f) <= node.t
            node = node.left;
        else
            node = node.right;
        end
    end
    prob(i,:) = node.p;
    pred(i) = argmax01(node.p)-1;
end
end

function rf = trainRF_multiclass_manual(X, y, C, nTrees, sampleFrac, maxDepth, nThreshTry, nFeatTry, minLeaf)
rf = struct();
rf.trees = cell(nTrees,1);
N = size(X,1);
nS = max(20, floor(sampleFrac*N));
for t=1:nTrees
    idx = randi(N, [nS,1]);
    rf.trees{t} = trainTree_multiclass_manual(X(idx,:), y(idx), C, maxDepth, nThreshTry, nFeatTry, minLeaf);
    if mod(t,10)==0
        fprintf('  RF tree %d/%d\n', t, nTrees);
    end
end
end

function [pred, prob] = predictRF_multiclass_manual(rf, X)
n = size(X,1);
T = numel(rf.trees);
C = numel(rf.trees{1}.p);
prob = zeros(n,C);
for t=1:T
    [~,p] = predictTree_multiclass_manual(rf.trees{t}, X);
    prob = prob + p;
end
prob = prob / T;
pred = zeros(n,1);
for i=1:n
    pred(i) = argmax01(prob(i,:))-1;
end
end

%% ----------------- SVM OVR (MANUAL) -------------------------------------
function [Wovr, bovr] = trainSVM_OVR_manual(X, y, C, epochs, lambda, eta0)
D = size(X,2);
Wovr = zeros(D,C);
bovr = zeros(1,C);
for c=0:C-1
    yy = -ones(size(y));
    yy(y==c) = 1;
    [W,b] = trainLinearSVM_SGD_manual(X, yy, epochs, lambda, eta0);
    Wovr(:,c+1) = W;
    bovr(c+1) = b;
end
end

function [W, b] = trainLinearSVM_SGD_manual(X, y, epochs, lambda, eta0)
[N,D] = size(X);
W = zeros(D,1);
b = 0.0;
tGlobal = 0;
for e=1:epochs
    idx = randperm(N);
    for t=1:N
        tGlobal = tGlobal + 1;
        i = idx(t);
        xi = X(i,:)';
        yi = y(i);
        eta = eta0 / (1 + lambda*eta0*tGlobal);
        margin = yi * (W' * xi + b);
        if margin < 1
            W = (1 - eta*lambda).*W + eta*yi.*xi;
            b = b + eta*yi;
        else
            W = (1 - eta*lambda).*W;
        end
    end
    fprintf('  SVM epoch %d/%d done\n', e, epochs);
end
end

function pred = svmPredict_OVR_manual(X, Wovr, bovr)
n = size(X,1);
C = size(Wovr,2);
pred = zeros(n,1);
for i=1:n
    bestC = 1;
    bestS = X(i,:)*Wovr(:,1) + bovr(1);
    for c=2:C
        s = X(i,:)*Wovr(:,c) + bovr(c);
        if s > bestS
            bestS = s;
            bestC = c;
        end
    end
    pred(i) = bestC-1;
end
end

function score = svmScore_binary_manual(X, Wovr, bovr)
score = X*Wovr(:,2) + bovr(2);
end

%% ----------------- METRICS + ROC ----------------------------------------
function printBinaryReportPlusROC(name, yTrue, yPred, scorePos)
yTrue = yTrue(:); yPred = yPred(:); scorePos = scorePos(:);

CM = zeros(2,2);
for i=1:numel(yTrue)
    CM(yTrue(i)+1, yPred(i)+1) = CM(yTrue(i)+1, yPred(i)+1) + 1;
end

acc = (CM(1,1)+CM(2,2)) / max(1,sum(CM(:)));

TP = CM(2,2);
FP = CM(1,2);
FN = CM(2,1);
prec = TP / (TP+FP+eps);
rec  = TP / (TP+FN+eps);
f1   = 2*prec*rec/(prec+rec+eps);

[~,~,auc] = roc_auc_manual(yTrue, scorePos);

fprintf('--- %s ---\n', name);
fprintf('Acc=%.4f  AUC=%.4f  F1(Building)=%.4f  Prec=%.4f  Rec=%.4f\n', acc, auc, f1, prec, rec);
fprintf('ConfMat (rows=true, cols=pred) [0=Road 1=Building]:\n');
disp(CM);
fprintf('\n');
end

function [fpr, tpr, auc] = roc_auc_manual(yTrue01, scorePos)
y = yTrue01(:);
s = scorePos(:);

[~, ord] = sort(s, 'descend');
yy = y(ord);

P = sum(yy==1);
N = sum(yy==0);
if P==0 || N==0
    fpr = [0;1]; tpr = [0;1]; auc = 0.5;
    return;
end

TP = 0; FP = 0;
tpr = zeros(numel(yy)+1,1);
fpr = zeros(numel(yy)+1,1);
tpr(1)=0; fpr(1)=0;

for i=1:numel(yy)
    if yy(i)==1
        TP = TP + 1;
    else
        FP = FP + 1;
    end
    tpr(i+1) = TP / P;
    fpr(i+1) = FP / N;
end

auc = 0;
for i=1:numel(fpr)-1
    auc = auc + (fpr(i+1)-fpr(i)) * (tpr(i+1)+tpr(i)) * 0.5;
end
end

%% ----------------- INFERENCE (BINARY) -----------------------------------
function predict_from_saved_model_binary( ...
    modelFile, imgPath, modelName, ...
    outLblSmall, outLblFull, outBldSmall, outBldFull, outRoadSmall, outRoadFull, ...
    outColSmall, outColFull, outOvSmall, outOvFull)

S = load(modelFile);
Iorig = imread(imgPath);
[H0,W0,~] = size(Iorig);

I = Iorig;
if isfield(S,'doDownsample') && S.doDownsample && isfield(S,'ds') && S.ds > 1
    ds = S.ds;
    I = Iorig(1:ds:end, 1:ds:end, :);
else
    ds = 1;
end
[H,W,~] = size(I);

F  = computeFeatures_noBuiltins_withMorph( ...
    I, S.morphRadius, ...
    S.doKMeansFeat, S.kmeansK, S.kmeansIters, S.kmeansSampleMax, ...
    S.doHoughFeat, S.houghThetas, S.houghTopN, S.houghNearPx);

Fz = zscore_manual(F, S.mu, S.sd);
if isfield(S,'doPCA') && S.doPCA && isfield(S,'P') && isfield(S,'kPCA')
    Fz = Fz * S.P(:,1:S.kPCA);
end

modelName = lower(modelName);

if strcmp(modelName,'knn')
    pred = knnPredict_multiclass_manual(S.XtrZ, S.ytr, Fz, S.kNN_k, 2);
elseif strcmp(modelName,'tree')
    [pred,~] = predictTree_multiclass_manual(S.tree, Fz);
elseif strcmp(modelName,'rf')
    [~,prob] = predictRF_multiclass_manual(S.rf, Fz);
    thrB = 0.65;                      % tune 0.60..0.80
    pred = uint8(prob(:,2) >= thrB);  % 1=building else 0=road
elseif strcmp(modelName,'svm')
    pred = svmPredict_OVR_manual(Fz, S.Wovr, S.bovr);
else
    error('Unknown modelName: %s', modelName);
end

lblSmall = uint8(reshape(pred, H, W)); % 0=Road, 1=Building

% ----------------- Post-proc (optional) ----------------------------------
if isfield(S,'doMorph') && S.doMorph
    se = makeDiskSE_manual(S.morphRadius);

    road = (lblSmall==0);
    bld  = (lblSmall==1);

    road = morph_close_open_manual(road, se);
    bld  = morph_close_open_manual(bld,  se);

    if isfield(S,'doRoadClean') && S.doRoadClean
        road = roadContinuityClean_manual(road, S.roadLineLen, S.minRoadSize);
    end
    if isfield(S,'doKeepTopKRoad') && S.doKeepTopKRoad
        road = keepTopKComponents_manual(road, S.keepTopKRoad);
    end
    if isfield(S,'doElongatedRoad') && S.doElongatedRoad
        road = keepElongatedComponents_manual(road, S.elongMinAspect, S.elongMinSize);
    end
    if isfield(S,'doRectangularBuildings') && S.doRectangularBuildings
        bld = keepRectangularComponents_manual(bld, S.rectMinFill, S.rectMinSize, S.rectMaxAspect);
    end

    road = road & ~bld; % building wins

    lblSmall = uint8(0)*lblSmall;
    lblSmall(road) = 0;
    lblSmall(bld)  = 1;
end

imwrite(lblSmall, outLblSmall);

bldSmall  = uint8(lblSmall==1)*255;
roadSmall = uint8(lblSmall==0)*255;
imwrite(bldSmall,  outBldSmall);
imwrite(roadSmall, outRoadSmall);

colSmall = labelToColorRGB_binary(lblSmall, S.visRoadColor, S.visBldColor);
imwrite(colSmall, outColSmall);

ovSmall = overlayBinary(I, lblSmall, S.visRoadColor, S.visBldColor, S.alphaOverlay);
imwrite(ovSmall, outOvSmall);

lblFull = upsample_label_nearest_manual(lblSmall, ds, H0, W0);
imwrite(lblFull, outLblFull);

bldFull  = uint8(lblFull==1)*255;
roadFull = uint8(lblFull==0)*255;
imwrite(bldFull,  outBldFull);
imwrite(roadFull, outRoadFull);

colFull = labelToColorRGB_binary(lblFull, S.visRoadColor, S.visBldColor);
imwrite(colFull, outColFull);

ovFull = overlayBinary(Iorig, lblFull, S.visRoadColor, S.visBldColor, S.alphaOverlay);
imwrite(ovFull, outOvFull);

pctB = 100*mean(lblFull(:)==1);
pctR = 100*mean(lblFull(:)==0);

figure('Name',['Prediction: ' imgPath],'NumberTitle','off');
subplot(1,3,1); imagesc(Iorig); axis image off; title('Input image','Interpreter','none');
subplot(1,3,2); imagesc(colFull); axis image off;
title(sprintf('Binary mask (FULL)\nRoad=%.1f%%  Building=%.1f%%', pctR, pctB),'Interpreter','none');
subplot(1,3,3); imagesc(ovFull); axis image off;
title(sprintf('Overlay (%s)\nRoad=Black  Building=Red', upper(modelName)),'Interpreter','none');

fprintf('  Predicted: %s\n', imgPath);
fprintf('    Road=%.2f%%  Building=%.2f%%\n', pctR, pctB);
end

function rgb = labelToColorRGB_binary(lbl, roadC, bldC)
lbl = uint8(lbl);
[H,W] = size(lbl);
rgb = zeros(H,W,3);
for i=1:H
    for j=1:W
        if lbl(i,j)==0
            c = roadC;
        else
            c = bldC;
        end
        rgb(i,j,1) = c(1);
        rgb(i,j,2) = c(2);
        rgb(i,j,3) = c(3);
    end
end
rgb = uint8(max(0,min(1,rgb))*255);
end

function out = overlayBinary(rgbImg, lbl, roadC, bldC, alpha)
img = double(rgbImg)/255.0;
lbl = uint8(lbl);
[H,W,~] = size(img);
col = zeros(H,W,3);
for i=1:H
    for j=1:W
        if lbl(i,j)==0
            c = roadC;
        else
            c = bldC;
        end
        col(i,j,1) = c(1);
        col(i,j,2) = c(2);
        col(i,j,3) = c(3);
    end
end
out = (1-alpha)*img + alpha*col;
out = uint8(max(0,min(1,out))*255);
end

function lblFull = upsample_label_nearest_manual(lblSmall, ds, H0, W0)
if ds <= 1
    lblFull = uint8(lblSmall);
    return;
end
[Hs,Ws] = size(lblSmall);
lblFull = zeros(H0,W0,'uint8');
for i=1:H0
    si = ceil(i/ds);
    si = min(max(si,1),Hs);
    for j=1:W0
        sj = ceil(j/ds);
        sj = min(max(sj,1),Ws);
        lblFull(i,j) = lblSmall(si,sj);
    end
end
end

%% -------- Keep only top-K largest connected components -------------------
function out = keepTopKComponents_manual(binMask, K)
A = logical(binMask);
[H,W] = size(A);
visited = false(H,W);
dirs = [ -1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1 ];
maxN = H*W;
sizes = [];
comps = {};
for i=1:H
    for j=1:W
        if ~A(i,j) || visited(i,j), continue; end
        stackI = zeros(maxN,1); stackJ = zeros(maxN,1);
        pix = zeros(maxN,1);
        top=1; stackI(top)=i; stackJ(top)=j;
        visited(i,j)=true;
        n=0;
        while top>0
            ci=stackI(top); cj=stackJ(top); top=top-1;
            n=n+1;
            pix(n) = sub2ind([H,W],ci,cj);
            for d=1:8
                ni=ci+dirs(d,1); nj=cj+dirs(d,2);
                if ni<1||ni>H||nj<1||nj>W, continue; end
                if A(ni,nj) && ~visited(ni,nj)
                    visited(ni,nj)=true;
                    top=top+1; stackI(top)=ni; stackJ(top)=nj;
                end
            end
        end
        sizes(end+1,1) = n; %#ok<AGROW>
        comps{end+1,1} = pix(1:n); %#ok<AGROW>
    end
end
out = false(H,W);
if isempty(sizes), return; end
[~,ord] = sort(sizes,'descend');
K = min(K, numel(ord));
for t=1:K
    idx = comps{ord(t)};
    out(idx) = true;
end
end

%% -------- Optional: keep only elongated components -----------------------
function out = keepElongatedComponents_manual(binMask, minAspect, minSize)
A = logical(binMask);
[H,W] = size(A);
visited = false(H,W);
dirs = [ -1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1 ];
maxN = H*W;
out = false(H,W);
for i=1:H
    for j=1:W
        if ~A(i,j) || visited(i,j), continue; end
        stackI = zeros(maxN,1); stackJ = zeros(maxN,1);
        pixI   = zeros(maxN,1); pixJ   = zeros(maxN,1);
        top=1; stackI(top)=i; stackJ(top)=j;
        visited(i,j)=true;
        n=0; minR=i; maxR=i; minC=j; maxC=j;
        while top>0
            ci=stackI(top); cj=stackJ(top); top=top-1;
            n=n+1;
            pixI(n)=ci; pixJ(n)=cj;
            if ci<minR, minR=ci; end
            if ci>maxR, maxR=ci; end
            if cj<minC, minC=cj; end
            if cj>maxC, maxC=cj; end
            for d=1:8
                ni=ci+dirs(d,1); nj=cj+dirs(d,2);
                if ni<1||ni>H||nj<1||nj>W, continue; end
                if A(ni,nj) && ~visited(ni,nj)
                    visited(ni,nj)=true;
                    top=top+1; stackI(top)=ni; stackJ(top)=nj;
                end
            end
        end
        if n >= minSize
            hBox = (maxR-minR+1);
            wBox = (maxC-minC+1);
            aspect = max(hBox,wBox) / max(1,min(hBox,wBox));
            if aspect >= minAspect
                for k=1:n
                    out(pixI(k), pixJ(k)) = true;
                end
            end
        end
    end
end
end

%% ----------------- ROAD CONTINUITY CLEAN --------------------------------
function road = roadContinuityClean_manual(road, lineLen, minSize)
road = logical(road);
se0   = makeLineSE_manual(lineLen, 0);
se90  = makeLineSE_manual(lineLen, 90);
se45  = makeLineSE_manual(lineLen, 45);
se135 = makeLineSE_manual(lineLen, 135);
road = close_manual(road, se0);
road = close_manual(road, se90);
road = close_manual(road, se45);
road = close_manual(road, se135);
seD = makeDiskSE_manual(2);
road = open_manual(road, seD);
road = removeSmallComponents_manual(road, minSize);
end

function se = makeLineSE_manual(len, angleDeg)
r = floor(len/2);
d = 2*r + 1;
se = false(d,d);
c = r + 1;
for t=-r:r
    if angleDeg == 0
        ii = c; jj = c + t;
    elseif angleDeg == 90
        ii = c + t; jj = c;
    elseif angleDeg == 45
        ii = c - t; jj = c + t;
    elseif angleDeg == 135
        ii = c + t; jj = c + t;
    else
        ii = c; jj = c + t;
    end
    if ii>=1 && ii<=d && jj>=1 && jj<=d
        se(ii,jj) = true;
    end
end
end

function out = close_manual(binMask, se)
out = erode_manual(dilate_manual(binMask, se), se);
end

function out = open_manual(binMask, se)
out = dilate_manual(erode_manual(binMask, se), se);
end

function out = removeSmallComponents_manual(binMask, minSize)
A = logical(binMask);
[H,W] = size(A);
out = false(H,W);
visited = false(H,W);
dirs = [ -1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1 ];
maxN = H*W;
for i=1:H
    for j=1:W
        if ~A(i,j) || visited(i,j), continue; end
        stackI = zeros(maxN,1); stackJ = zeros(maxN,1);
        compI  = zeros(maxN,1); compJ  = zeros(maxN,1);
        top = 1; stackI(top)=i; stackJ(top)=j;
        visited(i,j)=true;
        compN = 0;
        while top>0
            ci = stackI(top); cj = stackJ(top); top = top-1;
            compN = compN + 1;
            compI(compN)=ci; compJ(compN)=cj;
            for d=1:8
                ni = ci + dirs(d,1);
                nj = cj + dirs(d,2);
                if ni<1||ni>H||nj<1||nj>W, continue; end
                if A(ni,nj) && ~visited(ni,nj)
                    visited(ni,nj)=true;
                    top = top+1;
                    stackI(top)=ni; stackJ(top)=nj;
                end
            end
        end
        if compN >= minSize
            for k=1:compN
                out(compI(k), compJ(k)) = true;
            end
        end
    end
end
end

%% ----------------- MORPHOLOGY (NO TOOLBOX) ------------------------------
function se = makeDiskSE_manual(r)
d = 2*r + 1;
se = false(d,d);
cx = r + 1; cy = r + 1;
for i=1:d
    for j=1:d
        dx = i - cx;
        dy = j - cy;
        if (dx*dx + dy*dy) <= r*r
            se(i,j) = true;
        end
    end
end
end

function out = morph_close_open_manual(binMask, se)
close1 = erode_manual(dilate_manual(binMask, se), se);
out    = dilate_manual(erode_manual(close1, se), se);
end

function out = dilate_manual(A, se)
A = logical(A);
[H,W] = size(A);
[kH,kW] = size(se);
pH = floor(kH/2);
pW = floor(kW/2);
out = false(H,W);
for i=1:H
    for j=1:W
        hit = false;
        for a=1:kH
            ii = i + (a - (pH+1));
            if ii < 1 || ii > H, continue; end
            for b=1:kW
                if ~se(a,b), continue; end
                jj = j + (b - (pW+1));
                if jj < 1 || jj > W, continue; end
                if A(ii,jj), hit = true; break; end
            end
            if hit, break; end
        end
        out(i,j) = hit;
    end
end
end

function out = erode_manual(A, se)
A = logical(A);
[H,W] = size(A);
[kH,kW] = size(se);
pH = floor(kH/2);
pW = floor(kW/2);
out = false(H,W);
for i=1:H
    for j=1:W
        ok = true;
        for a=1:kH
            ii = i + (a - (pH+1));
            for b=1:kW
                if ~se(a,b), continue; end
                jj = j + (b - (pW+1));
                if ii < 1 || ii > H || jj < 1 || jj > W
                    ok = false; break;
                end
                if ~A(ii,jj)
                    ok = false; break;
                end
            end
            if ~ok, break; end
        end
        out(i,j) = ok;
    end
end
end

function out = grayDilate_manual(I, se)
[H,W] = size(I);
[kH,kW] = size(se);
pH = floor(kH/2); pW = floor(kW/2);
out = zeros(H,W);
for i=1:H
    for j=1:W
        mx = -inf;
        for a=1:kH
            ii = i + (a-(pH+1));
            if ii < 1 || ii > H, continue; end
            for b=1:kW
                if ~se(a,b), continue; end
                jj = j + (b-(pW+1));
                if jj < 1 || jj > W, continue; end
                v = I(ii,jj);
                if v > mx, mx = v; end
            end
        end
        if ~isfinite(mx), mx = I(i,j); end
        out(i,j) = mx;
    end
end
end

function out = grayErode_manual(I, se)
[H,W] = size(I);
[kH,kW] = size(se);
pH = floor(kH/2); pW = floor(kW/2);
out = zeros(H,W);
for i=1:H
    for j=1:W
        mn = inf;
        for a=1:kH
            ii = i + (a-(pH+1));
            if ii < 1 || ii > H, continue; end
            for b=1:kW
                if ~se(a,b), continue; end
                jj = j + (b-(pW+1));
                if jj < 1 || jj > W, continue; end
                v = I(ii,jj);
                if v < mn, mn = v; end
            end
        end
        if ~isfinite(mn), mn = I(i,j); end
        out(i,j) = mn;
    end
end
end

%% =================== KMEANS / HOUGH / LBP / HARRIS / TENSOR ==============
function [C, labels] = kmeans_manual_fit_predict(X, K, iters, sampleMax)
N = size(X,1); D = size(X,2);
nS = min(N, max(200, sampleMax));
idx = randperm(N, nS);
Xs = X(idx,:);
pick = randperm(nS, K);
C = Xs(pick,:);
labS = ones(nS,1);
for it=1:iters
    for i=1:nS
        bestK = 1; bestD2 = inf;
        for k=1:K
            d2 = 0.0;
            for j=1:D
                t = Xs(i,j) - C(k,j);
                d2 = d2 + t*t;
            end
            if d2 < bestD2
                bestD2 = d2; bestK = k;
            end
        end
        labS(i) = bestK;
    end
    Cnew = zeros(K,D);
    cnt = zeros(K,1);
    for i=1:nS
        k = labS(i);
        cnt(k) = cnt(k) + 1;
        for j=1:D
            Cnew(k,j) = Cnew(k,j) + Xs(i,j);
        end
    end
    for k=1:K
        if cnt(k) > 0
            Cnew(k,:) = Cnew(k,:) / cnt(k);
        else
            Cnew(k,:) = Xs(randi(nS),:);
        end
    end
    C = Cnew;
end
labels = ones(N,1);
for i=1:N
    bestK = 1; bestD2 = inf;
    for k=1:K
        d2 = 0.0;
        for j=1:D
            t = X(i,j) - C(k,j);
            d2 = d2 + t*t;
        end
        if d2 < bestD2
            bestD2 = d2; bestK = k;
        end
    end
    labels(i) = bestK;
end
end

function k = pickRoadCluster_manual(C)
K = size(C,1);
best = 1; bestScore = -inf;
for i=1:K
    S = C(i,2); V = C(i,3);
    score = (1 - S) - 0.8*abs(V - 0.55);
    if score > bestScore
        bestScore = score; best = i;
    end
end
k = best;
end

function d = sqDistToCentroid_manual(X, c)
N = size(X,1); D = size(X,2);
d = zeros(N,1);
for i=1:N
    s = 0.0;
    for j=1:D
        t = X(i,j) - c(j);
        s = s + t*t;
    end
    d(i) = s;
end
end

function lines = hough_lines_manual(E, thetasDeg, topN)
[H,W] = size(E);
ys = []; xs = [];
for i=1:H
    for j=1:W
        if E(i,j)
            ys(end+1,1) = i; %#ok<AGROW>
            xs(end+1,1) = j; %#ok<AGROW>
        end
    end
end
nP = numel(xs);
if nP == 0
    lines = struct('thetaDeg',{},'rho',{});
    return;
end
diagR = ceil(sqrt(H*H + W*W));
rhoBins = 2*diagR + 1;
allPeaks = [];
for tt=1:numel(thetasDeg)
    th = thetasDeg(tt) * pi/180;
    c = cos(th); s = sin(th);
    acc = zeros(rhoBins,1);
    for p=1:nP
        rho = round(xs(p)*c + ys(p)*s);
        rbin = rho + diagR + 1;
        if rbin>=1 && rbin<=rhoBins
            acc(rbin) = acc(rbin) + 1;
        end
    end
    for k=1:topN
        [mx,idx] = max(acc);
        if mx <= 0, break; end
        rho = idx - (diagR + 1);
        allPeaks(end+1,:) = [mx, thetasDeg(tt), rho]; %#ok<AGROW>
        acc(idx) = 0;
    end
end
if isempty(allPeaks)
    lines = struct('thetaDeg',{},'rho',{});
    return;
end
[~,ord] = sort(allPeaks(:,1),'descend');
ord = ord(1:min(topN,size(ord,1)));
lines = struct('thetaDeg',cell(numel(ord),1),'rho',cell(numel(ord),1));
for i=1:numel(ord)
    lines(i).thetaDeg = allPeaks(ord(i),2);
    lines(i).rho      = allPeaks(ord(i),3);
end
end

function score = lineProximityScore_manual(H, W, lines, nearPx)
score = zeros(H,W);
if isempty(lines), return; end
for i=1:H
    for j=1:W
        best = inf;
        for k=1:numel(lines)
            th = lines(k).thetaDeg*pi/180;
            rho = lines(k).rho;
            d = abs(j*cos(th) + i*sin(th) - rho);
            if d < best, best = d; end
        end
        if best <= nearPx
            score(i,j) = 1;
        else
            score(i,j) = exp(-(best-nearPx));
        end
    end
end
mx = max(score(:));
if mx > 0, score = score / mx; end
end

function [lbpMean, lbpVar] = lbp8_stats_manual(gray, win)
[H,W] = size(gray);
code = zeros(H,W);
for i=2:H-1
    for j=2:W-1
        c = gray(i,j);
        b0 = gray(i-1,j-1) >= c;
        b1 = gray(i-1,j  ) >= c;
        b2 = gray(i-1,j+1) >= c;
        b3 = gray(i  ,j+1) >= c;
        b4 = gray(i+1,j+1) >= c;
        b5 = gray(i+1,j  ) >= c;
        b6 = gray(i+1,j-1) >= c;
        b7 = gray(i  ,j-1) >= c;
        v = b0*1 + b1*2 + b2*4 + b3*8 + b4*16 + b5*32 + b6*64 + b7*128;
        code(i,j) = v;
    end
end
k = ones(win,win) / (win*win);
m  = conv2_manual(code, k);
m2 = conv2_manual(code.^2, k);
v  = m2 - m.^2;
v(v<0)=0;
lbpMean = m / 255;
lbpVar  = v / (255*255);
end

function corner = harrisCornerScore_manual(Gx, Gy, win, kappa)
Gx2 = Gx.^2;
Gy2 = Gy.^2;
Gxy = Gx.*Gy;
k = ones(win,win) / (win*win);
Sx2 = conv2_manual(Gx2, k);
Sy2 = conv2_manual(Gy2, k);
Sxy = conv2_manual(Gxy, k);
[H,W] = size(Gx);
corner = zeros(H,W);
for i=1:H
    for j=1:W
        a = Sx2(i,j);
        b = Sxy(i,j);
        c = Sy2(i,j);
        detM = a*c - b*b;
        trM  = a + c;
        corner(i,j) = detM - kappa*(trM*trM);
    end
end
corner(corner<0)=0;
mx = max(corner(:));
if mx>0, corner = corner/mx; end
end

function lin = structureTensorLinearity_manual(Gx, Gy, win)
Gx2 = Gx.^2;
Gy2 = Gy.^2;
Gxy = Gx.*Gy;
k = ones(win,win) / (win*win);
Sx2 = conv2_manual(Gx2, k);
Sy2 = conv2_manual(Gy2, k);
Sxy = conv2_manual(Gxy, k);
[H,W] = size(Gx);
lin = zeros(H,W);
for i=1:H
    for j=1:W
        a = Sx2(i,j);
        b = Sxy(i,j);
        c = Sy2(i,j);
        tr = a + c;
        disc = (a-c)*(a-c) + 4*b*b;
        rt = sqrt(max(0,disc));
        l1 = 0.5*(tr + rt);
        l2 = 0.5*(tr - rt);
        lin(i,j) = (l1 - l2) / (l1 + l2 + 1e-12);
    end
end
lin(lin<0)=0; lin(lin>1)=1;
end

function out = keepRectangularComponents_manual(binMask, minFill, minSize, maxAspect)
A = logical(binMask);
[H,W] = size(A);
visited = false(H,W);
dirs = [ -1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1 ];
maxN = H*W;
out = false(H,W);
for i=1:H
    for j=1:W
        if ~A(i,j) || visited(i,j), continue; end
        stackI = zeros(maxN,1); stackJ = zeros(maxN,1);
        pixI   = zeros(maxN,1); pixJ   = zeros(maxN,1);
        top=1; stackI(top)=i; stackJ(top)=j;
        visited(i,j)=true;
        n=0; minR=i; maxR=i; minC=j; maxC=j;
        while top>0
            ci=stackI(top); cj=stackJ(top); top=top-1;
            n=n+1;
            pixI(n)=ci; pixJ(n)=cj;
            if ci<minR, minR=ci; end
            if ci>maxR, maxR=ci; end
            if cj<minC, minC=cj; end
            if cj>maxC, maxC=cj; end
            for d=1:8
                ni=ci+dirs(d,1); nj=cj+dirs(d,2);
                if ni<1||ni>H||nj<1||nj>W, continue; end
                if A(ni,nj) && ~visited(ni,nj)
                    visited(ni,nj)=true;
                    top=top+1; stackI(top)=ni; stackJ(top)=nj;
                end
            end
        end
        if n >= minSize
            hBox = (maxR-minR+1);
            wBox = (maxC-minC+1);
            bboxArea = hBox*wBox;
            fill = n / max(1,bboxArea);
            aspect = max(hBox,wBox) / max(1,min(hBox,wBox));
            if (fill >= minFill) && (aspect <= maxAspect)
                for k=1:n
                    out(pixI(k), pixJ(k)) = true;
                end
            end
        end
    end
end
end
%% ============================ PLOTTING HELPERS ===========================

function plotClassProportions_binary(yTrue, predList, modelNames)
% yTrue: Nx1 (0 road, 1 building)
% predList: cell array {predModel1, predModel2, ...}
% modelNames: cell array of strings

gtRoad = 100*mean(yTrue==0);
gtBld  = 100*mean(yTrue==1);

M = numel(predList);
roadP = zeros(1, M+1);
bldP  = zeros(1, M+1);

roadP(1) = gtRoad; 
bldP(1)  = gtBld;

for i=1:M
    roadP(i+1) = 100*mean(predList{i}==0);
    bldP(i+1)  = 100*mean(predList{i}==1);
end

labs = [{'GT'}, modelNames(:)'];
Y = [roadP; bldP]';  % (M+1) x 2

figure('Name','Road vs Building Proportions','NumberTitle','off');
bar(Y);
set(gca,'XTick',1:numel(labs),'XTickLabel',labs);
ylabel('Percentage (%)');
title('Road vs Building Proportions (Sampled Pixels)');
legend({'Road','Building'},'Location','best');
grid on;
end


function plotConfMatHeatmap_binary(yTrue, yPred, figTitle)
CM = zeros(2,2);
for i=1:numel(yTrue)
    CM(yTrue(i)+1, yPred(i)+1) = CM(yTrue(i)+1, yPred(i)+1) + 1;
end

figure('Name',figTitle,'NumberTitle','off');
imagesc(CM);
axis image;
colorbar;
title(figTitle);
xlabel('Predicted'); ylabel('True');
set(gca,'XTick',[1 2],'XTickLabel',{'Road(0)','Bld(1)'});
set(gca,'YTick',[1 2],'YTickLabel',{'Road(0)','Bld(1)'});

% numbers on cells
for r=1:2
    for c=1:2
        text(c,r,sprintf('%d',CM(r,c)), ...
            'Color','w','FontSize',12,'FontWeight','bold', ...
            'HorizontalAlignment','center');
    end
end
end


function plotRGBHistograms_binary(Xtr, ytr, nBins)
% Uses your feature order: R=6, G=7, B=8 in Xtr
R = Xtr(:,6); 
G = Xtr(:,7); 
B = Xtr(:,8);

figure('Name','RGB channel histograms (Road vs Building)','NumberTitle','off');

subplot(3,1,1);
plotOneHist(R, ytr, nBins, 'R channel histogram (Road vs Building)', 'R intensity');

subplot(3,1,2);
plotOneHist(G, ytr, nBins, 'G channel histogram (Road vs Building)', 'G intensity');

subplot(3,1,3);
plotOneHist(B, ytr, nBins, 'B channel histogram (Road vs Building)', 'B intensity');
end


function plotOneHist(vec, y, nBins, ttl, xlab)
road = vec(y==0);
bld  = vec(y==1);

mn = min(vec); mx = max(vec);
if mx <= mn, mx = mn + 1e-6; end
edges = linspace(mn, mx, nBins+1);
centres = 0.5*(edges(1:end-1) + edges(2:end));

hR = histcounts(road, edges);
hB = histcounts(bld,  edges);

plot(centres, hR, '-'); hold on;
plot(centres, hB, '-'); hold off;

title(ttl);
xlabel(xlab);
ylabel('Count');
legend({'Road','Building'},'Location','best');
grid on;
end

%% ============================ DASHBOARD (MAIN PLOTS) ============================
function plotDashboard6_binary( ...
    Xtr, ytr, ...
    yte, ...
    predKNN, predTree, predRF, predSVM, ...
    scoreKNN, scoreTree, scoreRF, scoreSVM, ...
    featNames, ...
    showPR)
% ---- Safety checks ----
if isempty(yte) || isempty(predRF) || numel(yte) ~= numel(predRF)
    warning('Dashboard skipped: yte/pred arrays empty or mismatched.');
    return;
end
figure('Name','Dashboard (Road vs Building)','NumberTitle','off','Color','w');
set(gcf,'Renderer','painters');

tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

% Preallocate legend cell arrays
lg  = cell(1,4);
lg2 = cell(1,4);

%% (1) Grayscale Pixel Intensity Distribution (TRAIN)
nexttile;   % ✅ IMPORTANT: start first tile explicitly
histogram(Xtr(:,1), 50, 'Normalization','pdf');   % Feature 1 = gray
xlabel('Grayscale Intensity');
ylabel('Probability Density');
title('1) Grayscale Pixel Intensity (Train)');
grid on;

%% (2) Correlation matrix (TRAIN features)
nexttile;
Xs = subsampleRows(Xtr, 6000);
Xs = Xs(all(isfinite(Xs),2),:);    % remove NaN/Inf rows
if size(Xs,1) < 5
    R = zeros(size(Xtr,2));
else
    R = corrcoef(Xs);
end
R(~isfinite(R)) = 0;               % constant columns => NaN correlations

imagesc(R); axis image;
set(gca,'YDir','normal');
colorbar;
title('2) Correlation matrix (Train features)');

D = size(Xtr,2);
if ~isempty(featNames) && numel(featNames)==D
    step = max(1, round(D/8));
    ticks = 1:step:D;
    xticks(ticks); yticks(ticks);
    xticklabels(featNames(ticks));
    yticklabels(featNames(ticks));
end

%% (3) Class proportions (GT + each model)
nexttile;
names = {'GT','KNN','Tree','RF','SVM'};
preds = {yte, predKNN, predTree, predRF, predSVM};

roadPct = zeros(1,numel(preds));
bldPct  = zeros(1,numel(preds));
for i=1:numel(preds)
    roadPct(i) = 100*mean(preds{i}==0);
    bldPct(i)  = 100*mean(preds{i}==1);
end

bar([roadPct(:) bldPct(:)], 'grouped');
title('3) Class proportions (%)');
ylabel('%');
xticks(1:numel(names)); xticklabels(names);
legend({'Road','Building'},'Location','best');
grid on;

%% (4) Precision–Recall curves (OPTIONAL)
nexttile;
if showPR
    hold on;
    [P,Rr,ap] = pr_curve_manual(yte, scoreKNN);  plot(Rr,P); lg{1}=sprintf('KNN AP=%.3f',ap);
    [P,Rr,ap] = pr_curve_manual(yte, scoreTree); plot(Rr,P); lg{2}=sprintf('Tree AP=%.3f',ap);
    [P,Rr,ap] = pr_curve_manual(yte, scoreRF);   plot(Rr,P); lg{3}=sprintf('RF AP=%.3f',ap);
    [P,Rr,ap] = pr_curve_manual(yte, scoreSVM);  plot(Rr,P); lg{4}=sprintf('SVM AP=%.3f',ap);
    title('4) Precision–Recall (Building=Positive)');
    xlabel('Recall'); ylabel('Precision');
    legend(lg,'Location','SouthWest');
    grid on;
    hold off;
else
    axis off;
    text(0.05,0.6,'4) Precision–Recall disabled','FontWeight','bold');
    text(0.05,0.4,'Set showPR=true to enable it.');
end

%% (5) RF confusion matrix heatmap
nexttile;
CM = zeros(2,2);
for i=1:numel(yte)
    CM(yte(i)+1, predRF(i)+1) = CM(yte(i)+1, predRF(i)+1) + 1;
end
imagesc(CM);
axis image;
set(gca,'YDir','normal');
colorbar;
title('5) RF Confusion Matrix');
xlabel('Predicted'); ylabel('True');
xticks([1 2]); yticks([1 2]);
xticklabels({'Road(0)','Bld(1)'}); yticklabels({'Road(0)','Bld(1)'});

for r=1:2
    for c=1:2
        text(c,r,sprintf('%d',CM(r,c)), ...
            'HorizontalAlignment','center', ...
            'Color','w','FontWeight','bold');
    end
end

%% (6) ROC curves
nexttile;
hold on;
[fpr,tpr,auc] = roc_auc_manual(yte, scoreKNN);  plot(fpr,tpr); lg2{1}=sprintf('KNN AUC=%.3f',auc);
[fpr,tpr,auc] = roc_auc_manual(yte, scoreTree); plot(fpr,tpr); lg2{2}=sprintf('Tree AUC=%.3f',auc);
[fpr,tpr,auc] = roc_auc_manual(yte, scoreRF);   plot(fpr,tpr); lg2{3}=sprintf('RF AUC=%.3f',auc);
[fpr,tpr,auc] = roc_auc_manual(yte, scoreSVM);  plot(fpr,tpr); lg2{4}=sprintf('SVM AUC=%.3f',auc);
plot([0 1],[0 1],'k--');
title('6) ROC curves (Building=Positive)');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
legend(lg2,'Location','SouthEast');
grid on;
hold off;

drawnow;
end
function [prec, rec, ap] = pr_curve_manual(yTrue01, scorePos)
% yTrue01: {0,1}, scorePos: higher = more likely positive (building)

y = yTrue01(:);
s = scorePos(:);

% sort by score descending
[~, ord] = sort(s, 'descend');
y = y(ord);

P = sum(y==1);
if P == 0
    prec = 1; rec = 0; ap = 0;
    return;
end

tp = 0; fp = 0;
N = numel(y);

prec = zeros(N,1);
rec  = zeros(N,1);

for i=1:N
    if y(i)==1
        tp = tp + 1;
    else
        fp = fp + 1;
    end
    prec(i) = tp / (tp + fp + eps);
    rec(i)  = tp / (P + eps);
end

% Average Precision (AP) via step-wise integral
ap = 0;
rPrev = 0;
for i=1:N
    if rec(i) > rPrev
        ap = ap + prec(i) * (rec(i) - rPrev);
        rPrev = rec(i);
    end
end
end

function Xs = subsampleRows(X, maxN)
% Randomly subsample rows for faster plotting
N = size(X,1);
if N <= maxN
    Xs = X;
else
    idx = randperm(N, maxN);
    Xs = X(idx,:);
end
end


function ok = tryDownload(url, outFile)
ok = false;

try
    if exist('websave','file') == 2
        opts = weboptions('Timeout',180,'UserAgent','Mozilla/5.0');
        websave(outFile, url, opts);
        ok = exist(outFile,'file')==2 && dir(outFile).bytes > 0;
        if ok, return; end
    end
catch
end

% fallback to curl (handles GitHub redirects)
try
    if ispc
        cmd = sprintf('curl -L -o "%s" "%s"', outFile, url);
    else
        cmd = sprintf('curl -L -o ''%s'' ''%s''', outFile, url);
    end
    status = system(cmd);
    ok = (status==0) && exist(outFile,'file')==2 && dir(outFile).bytes > 0;
catch
    ok = false;
end
end


function [imgDir, mskDir] = ensureDataset(IMG_DATA, SATIMG, imgDir, mskDir)
% ensureDataset
% Downloads + unzips dataset into SATIMG if images/masks not already present.

    % If already present, skip download
    if exist(imgDir,'dir') && exist(mskDir,'dir')
        fprintf('[DATA] Found existing dataset folders.\n');
        return;
    end

    if ~exist(SATIMG,'dir')
        mkdir(SATIMG);
    end

    % Always write ZIP to a LOCAL file path
    zipPath = fullfile(SATIMG, 'Imgdataset.zip');

    fprintf('\n[DATA] Downloading Imgdataset.zip ...\n');
    ok = tryDownload(IMG_DATA, zipPath);
    if ~ok
        error('Download failed from: %s', IMG_DATA);
    end

    fprintf('[DATA] Unzipping into: %s\n', SATIMG);
    unzip(zipPath, SATIMG);

    % First try expected structure
    candImg = fullfile(SATIMG, 'images');
    candMsk = fullfile(SATIMG, 'masks');

    if exist(candImg,'dir') && exist(candMsk,'dir')
        imgDir = candImg;
        mskDir = candMsk;
        fprintf('[DATA] Using:\n  IMG_DIR = %s\n  MASK_DIR = %s\n', imgDir, mskDir);
        return;
    end

    % If GitHub zip contains one inner folder, search inside it
    L = dir(SATIMG);
    L = L([L.isdir] & ~ismember({L.name},{'.','..'}));

    for k = 1:numel(L)
        inner = fullfile(SATIMG, L(k).name);
        candImg = fullfile(inner, 'images');
        candMsk = fullfile(inner, 'masks');

        if exist(candImg,'dir') && exist(candMsk,'dir')
            imgDir = candImg;
            mskDir = candMsk;
            fprintf('[DATA] Using:\n  IMG_DIR = %s\n  MASK_DIR = %s\n', imgDir, mskDir);
            return;
        end
    end

    error('After unzip, images/ and masks/ folders were not found under %s.', SATIMG);
end
