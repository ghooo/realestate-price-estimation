#! /usr/bin/octave -qf
1;
function [avg_green, avg_blue, vertical_brightness] = getFeatures(dataset_dir, images)
	avg_green = zeros(length(images), 1);
	avg_blue = zeros(length(images), 1);
	vertical_brightness = zeros(length(images), 1);
	for i = 1:1:length(images)
		inimg = imread([dataset_dir images(i).name]);
		upperThird_rows = 1:1:round(rows(inimg)/3);
		lowerThird_rows = round(rows(inimg)-rows(inimg)/3):1:rows(inimg);
		avg_green(i, 1) = sum(inimg(lowerThird_rows, :, 2)(:)) / (length(inimg) * length(lowerThird_rows));
		% avg_green(i, 1) = avg_green(i, 1) / (sum((inimg(lowerThird_rows, :, 1) + inimg(lowerThird_rows, :, 2) + inimg(lowerThird_rows, :, 3))(:))/3/(length(inimg) * length(lowerThird_rows)));
		% if(avg_green(i, 1) / (sum((inimg(lowerThird_rows, :, 1) + inimg(lowerThird_rows, :, 2) + inimg(lowerThird_rows, :, 3))(:))/3/(length(inimg) * length(upperThird_rows))) <= 1.2)
		% 	avg_green(i, 1) = 0;
		% end
		avg_blue(i, 1) = sum(inimg(upperThird_rows, :, 3)(:)) / (length(inimg) * length(upperThird_rows));
		upper_brightness = sum((inimg(upperThird_rows, :, 1) + inimg(upperThird_rows, :, 2) + inimg(upperThird_rows, :, 3))(:))/3/(length(inimg) * length(upperThird_rows));
		lower_brightness = sum((inimg(lowerThird_rows, :, 1) + inimg(lowerThird_rows, :, 2) + inimg(lowerThird_rows, :, 3))(:))/3/(length(inimg) * length(lowerThird_rows));
		vertical_brightness(i, 1) = upper_brightness - lower_brightness;
	end
end

function cnt_positives = classify(features, features_idx, c1, c2, mu1, mu2, covar1_inv, covar2_inv)
	cnt_positives = 0;
	for i = 1:1:rows(features)
		gauss_score1 = c1*exp(-0.5*(features(i, features_idx) - mu1) * covar1_inv * (features(i, features_idx) - mu1)');
		gauss_score2 = c2*exp(-0.5*(features(i, features_idx) - mu2) * covar2_inv * (features(i, features_idx) - mu2)');
		if(gauss_score1 > gauss_score2)
			cnt_positives = cnt_positives + 1;
		end
	end
end

outdoor_dataset_dir = "dataset_outdoors/";
outdoor_images_directories = dir(outdoor_dataset_dir);
outdoor_images_directories([1 2]) = [];

indoor_dataset_dir = "dataset_indoors/";
indoor_images_directories = dir(indoor_dataset_dir);
indoor_images_directories([1 2]) = [];

num_testingSet_outdoor = round(length(outdoor_images_directories)/5);
num_testingSet_indoor = round(length(indoor_images_directories)/5);

trainingSetImages_outdoor = outdoor_images_directories(1:1:length(outdoor_images_directories)-num_testingSet_outdoor-1);
testingSetImages_outdoor = outdoor_images_directories(length(outdoor_images_directories)-num_testingSet_outdoor:1:length(outdoor_images_directories));

trainingSetImages_indoor = indoor_images_directories(1:1:length(indoor_images_directories)-num_testingSet_indoor-1);
testingSetImages_indoor = indoor_images_directories(length(indoor_images_directories)-num_testingSet_indoor:1:length(indoor_images_directories));

% printf("Starting feature extraction: outdoor\n");
[avg_green_outdoor, avg_blue_outdoor, vertical_brightness_outdoor] = getFeatures(outdoor_dataset_dir, trainingSetImages_outdoor);
% printf("Starting feature extraction: indoor\n");
[avg_green_indoor, avg_blue_indoor, vertical_brightness_indoor] = getFeatures(indoor_dataset_dir, trainingSetImages_indoor);
% printf("Starting feature extraction: testing set outdoor\n");
[avg_green_outdoor_test, avg_blue_outdoor_test, vertical_brightness_outdoor_test] = getFeatures(outdoor_dataset_dir, testingSetImages_outdoor);
% printf("Starting feature extraction: testing set indoor\n");
[avg_green_indoor_test, avg_blue_indoor_test, vertical_brightness_indoor_test] = getFeatures(indoor_dataset_dir, testingSetImages_indoor);
% printf("Done feature extraction.\n");

features_idx = [1];
features_outdoor = [avg_green_outdoor avg_blue_outdoor vertical_brightness_outdoor];
features_indoor = [avg_green_indoor avg_blue_indoor vertical_brightness_indoor];
mu_outdoor = mean(features_outdoor(:, features_idx));
mu_indoor = mean(features_indoor(:, features_idx));
covar_outdoor = cov(features_outdoor(:, features_idx));
covar_indoor = cov(features_indoor(:, features_idx));
const_outdoor = 1/((2*pi)^(length(features_idx)/2) * det(covar_outdoor)^0.5);
const_indoor = 1/((2*pi)^(length(features_idx)/2) * det(covar_indoor)^0.5);
covar_outdoor_inv = pinv(covar_outdoor);
covar_indoor_inv = pinv(covar_indoor);
features_test_outdoor = [avg_green_outdoor_test avg_blue_outdoor_test vertical_brightness_outdoor_test];
features_test_indoor = [avg_green_indoor_test avg_blue_indoor_test vertical_brightness_indoor_test];

cnt_outdoor = classify(features_test_outdoor, features_idx, const_outdoor, const_indoor, mu_outdoor, mu_indoor, covar_outdoor_inv, covar_indoor_inv);
cnt_indoor = classify(features_test_indoor, features_idx, const_outdoor, const_indoor, mu_outdoor, mu_indoor, covar_outdoor_inv, covar_indoor_inv);

outdoor_accuracy = cnt_outdoor/length(testingSetImages_outdoor)*100
indoor_accuracy = cnt_indoor/length(testingSetImages_indoor)*100
overall_accuracy = (cnt_outdoor + cnt_indoor)/(length(testingSetImages_outdoor) + length(testingSetImages_indoor))*100


% width = 10;
% figure(1);
% temp = histc(vertical_brightness_outdoor, -20:40/20:20);
% temp = temp ./length(trainingSetImages_outdoor);
% bar(temp);
% figure(2);
% temp = histc(vertical_brightness_indoor, -20:40/20:20);
% temp = temp ./length(trainingSetImages_outdoor);
% bar(temp);
% figure(3);
% temp = histc(avg_green_outdoor, 0:width:255);
% temp = temp ./length(trainingSetImages_outdoor);
% bar(temp);
% figure(4);
% temp = histc(avg_green_indoor, 0:width:255);
% temp = temp ./length(trainingSetImages_outdoor);
% bar(temp);
% figure(5);
% temp = histc(avg_blue_outdoor, 0:width:255);
% temp = temp ./length(trainingSetImages_outdoor);
% bar(temp);
% figure(6);
% temp = histc(avg_blue_indoor, 0:width:255);
% temp = temp ./length(trainingSetImages_outdoor);
% bar(temp);
% pause
