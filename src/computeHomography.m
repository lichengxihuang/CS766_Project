function H_3x3 = computeHomography(src_pts_nx2, dest_pts_nx2)

num_pairs = size(src_pts_nx2, 1);

A = [];

for i = 1: num_pairs
    xs = src_pts_nx2(i, 1);
    ys = src_pts_nx2(i, 2);
    
    xd = dest_pts_nx2(i, 1);
    yd = dest_pts_nx2(i, 2);
    
    curr = [
        xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd;
        0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd
        ];
    
    A = vertcat(A, curr);
end

% e = eig(A'*A)
[V,D] = eig(A'*A);

[lambda, i] = min(diag(D));

% lambda * V(:,i) - A'*A*V(:,i)
% V(:,i);
H_3x3 = reshape(V(:,i), [3,3])';


