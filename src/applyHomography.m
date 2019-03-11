function dest_pts_nx2 = applyHomography(H_3x3, src_pts_nx2)
num_pairs = size(src_pts_nx2, 1);

dest = H_3x3 * vertcat(src_pts_nx2', ones(1,num_pairs));

dest_pts_nx2 = [
    dest(1,:) ./ dest(3,:);
    dest(2,:) ./ dest(3,:)
    ]';
