clc
clear all

% Loading a built in dataset
images = imageDatastore('MerchData', 'IncludeSubfolders',true, 'LabelSource','foldernames');

% Splitting the dataset
[trainingImages, testImages] = splitEachLabel(images,0.7,'randomized');

% Showing some of the images in figure
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages, 9);
figure
for i = 1:9
    subplot(3,3,i)
    I = readimage(trainingImages, idx(i));
    imshow(I)
    label = trainingImages.Labels(idx(i));
    title(char(label))
end

% Loading Network
net = alexnet;

% Loading all layers of AlexNet except the last 3 ones
layersTransfer = net.Layers(1:end-3);

% Adding new layer add end of extracted layer having numClasses
numClasses = numel(categories(trainingImages.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Specifying training parameters
miniBatchSize = 5;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',true,...
    'Plots','training-progress',...
    'ValidationData',testImages,...
    'ValidationFrequency',numIterationsPerEpoch);

% Training the network
netTransfer = trainNetwork(trainingImages,layers,options);

% Prediciting the output labels on test set
predictedLabels = classify(netTransfer,testImages);

% Printing some of the output Images
numTestImages = numel(testImages.Labels);
idx = randperm(numTestImages, 16);
figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(testImages, idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

% Getting Accuracy of classifier
testLabels = testImages.Labels;
accuracy = mean(predictedLabels == testLabels);
disp(strcat('The test Accuracy of Classifier is:', num2str(accuracy*100), '%'))