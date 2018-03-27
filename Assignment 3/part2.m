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

% Extracting fc7 features for train and test data
layer = 'fc7';
trainingFeatures = activations(net, trainingImages, layer);
testFeatures = activations(net, testImages, layer);

% Getting train and test Labels
trainingLabels = trainingImages.Labels;
testLabels = testImages.Labels;

% Creating an SVM classifier for these features
classifier = fitcecoc(trainingFeatures, trainingLabels);

% Predicting Labels from the classifier of the test Images
predictedLabels = predict(classifier, testFeatures);

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
accuracy = mean(predictedLabels == testLabels);
disp(strcat('The test Accuracy of Classifier is:', num2str(accuracy*100), '%'))