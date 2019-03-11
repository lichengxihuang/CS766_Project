function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H,...
    dest_canvas_width_height)

width = dest_canvas_width_height(1);
height = dest_canvas_width_height(2);

result_img = zeros(height, width, 3);

[X_src,Y_src] = meshgrid(1:size(src_img,2), 1:size(src_img,1));
[X_dest,Y_dest] = meshgrid(1:width, 1:height);

res = resultToSrc_H * [X_dest(:),Y_dest(:),ones(numel(X_dest),1)]';


xs = res(1,:) ./ res(3,:);
ys = res(2,:) ./ res(3,:);

xs = reshape(xs, [height, width]);
ys = reshape(ys, [height, width]);

result_img(:,:,1) = interp2(X_src, Y_src, src_img(:,:,1), xs, ys);
result_img(:,:,2) = interp2(X_src, Y_src, src_img(:,:,2), xs, ys);
result_img(:,:,3) = interp2(X_src, Y_src, src_img(:,:,3), xs, ys);

% imshow(result_img);
mask3 = ~isnan(result_img);
mask = mask3(:,:,1) & mask3(:,:,2) & mask3(:,:,3);

result_img(~mask3) = 0;
