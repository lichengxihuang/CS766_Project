I1 = rgb2gray(imread('../dataset/stopsign_o.png'));
I2 = rgb2gray(imread('../dataset/stopsign13.jpg'));
% I2 = imsh

% corners1 = detectHarrisFeatures(I1);
% corners2 = detectHarrisFeatures(I2);
points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);

[f1, vpts1] = extractFeatures(I1, points1);
[f2, vpts2] = extractFeatures(I2, points2);

indexPairs = matchFeatures(f1, f2, 'Unique', true);

matched_points1 = vpts1(indexPairs(:, 1));
matched_points2 = vpts2(indexPairs(:, 2));


% [inliers_id, H] = runRANSAC(matched_points1.Location, matched_points2.Location, 100, 20);
% matched_points1 = matched_points1(inliers_id);
% matched_points2 = matched_points2(inliers_id);


figure; showMatchedFeatures(I1, I2, matched_points1, matched_points2, 'montage');
[tform, inlier_points1, inlier_points2] = estimateGeometricTransform(matched_points1, matched_points2, 'affine');
figure; showMatchedFeatures(I1, I2, inlier_points1, inlier_points2, 'montage');





polygon = [1, 1;...                           % top-left
        size(I1, 2), 1;...                 % top-right
        size(I1, 2), size(I1, 1);... % bottom-right
        1, size(I1, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon

newPolygon = transformPointsForward(tform, polygon);


figure;
imshow(I2);
hold on;
line(newPolygon(:, 1), newPolygon(:, 2), 'Color', 'b','LineWidth', 2);



