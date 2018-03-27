clc
clear all

disp('Loading AlexNet....')
net = alexnet;
disp('Network Loaded.')
disp(net);

disp('Printing layers of AlexNet....')
disp(net.Layers);

disp('Printing 10 classes of AlexNet.')
disp(net.Layers(end).ClassNames(100:110));

disp('Reading Image')
img = imread('car_image.png');
img = imresize(img, [227 227]);

disp('Classifying Image')
label = classify(net, img);

disp(strcat('The input Image belongs to the class: ', char(label)));

figure, imshow(img)
title(char(label))