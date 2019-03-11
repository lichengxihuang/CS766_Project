function [inliers_id, H] = runRANSAC(Xs, Xd, ransac_n, eps)

H = zeros(3,3);
inliers_id = [];
max_match = 0;

for i = 1:ransac_n
    sample_i = randperm(size(Xs, 1),4);
    
    src_pts = Xs(sample_i,:);
    dest_pts = Xd(sample_i,:);
    
    curr_H = computeHomography(src_pts, dest_pts);
    new_Xd = applyHomography(curr_H, Xs);
    diff = (Xd - new_Xd);
    dist = sqrt(diff(:,1).^2 + diff(:,2).^2);
    
    curr_inliers_id = find(dist < eps);
    if length(curr_inliers_id) > max_match
        H = curr_H;
        inliers_id = curr_inliers_id;
        max_match = length(curr_inliers_id);
    end
end
