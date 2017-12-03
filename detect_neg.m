% Takes in a nxd matrix X, 
% runs locality-sensitive hashing to detect the negative entries.
% Returns a list of indexes in the original C to signal the indices of 
% negative in C=XX'.
% 
% Eg. X = [ -1  1
%            1 -1
%            1  1 ]
% C = XX' = [ 2 -2 0
%             -2 2 0
%             0  0 2 ]
% Returns linear indices

% Runs in sub-quadratic time, so only find a subset of the negative entries.
function mask = detect_neg(X, r, k)
X_norm = normr(X);  % TODO: see if we can enforce this as precondition
d = size(X,2);
[a, b] = gen_ab(d, r, k);
[tr_hash_vals, tr_idx_map, tr_meta_map] = lsh(X_norm, a, b, r);
[tt_hash_vals, tt_idx_map, tt_meta_map] = lsh(-1 * X_norm, a, b, r);
fprintf("Training hash table size: %i\n. ", length(tr_idx_map));
disp(get_bin_sizes(tr_meta_map));
fprintf("Testing hash table size: %i\n", length(tt_idx_map));
disp(get_bin_sizes(tt_meta_map));
mask = find_neg_entries(X, tr_idx_map, tr_meta_map, tt_idx_map, tt_meta_map);
mask = unique(mask);
end

function [binsizes] = get_bin_sizes(meta_map)
binsizes = zeros(1,length(meta_map));
ks = keys(meta_map);
for i = 1:length(meta_map)
    temp = meta_map(ks{i});
    binsizes(i) = temp(2);
end
end

function [mask] = find_neg_entries(X, tr_idx_map, tr_meta_map, tt_idx_map, tt_meta_map)
N = size(X , 1);
tt_keys = keys(tt_idx_map);
mask = zeros(1, 200); ptr = 1; len=200;
for i = 1:length(tt_idx_map)
    hash_val = tt_keys{i};
    if isKey(tr_idx_map, hash_val) == 0
        continue
    end
    tr_len = tr_meta_map(hash_val); tr_len = tr_len(2) - 1;
    tt_len = tt_meta_map(hash_val); tt_len = tt_len(2) - 1;
    tr_ar = tr_idx_map(hash_val); tr_ar = tr_ar(1:tr_len);
    tt_ar = tt_idx_map(hash_val); tt_ar = tt_ar(1:tt_len);
    
    Xtr = X(tr_ar, :); Xtt = X(tt_ar, :);
    C = Xtr * Xtt';
    lin_id = find(C < 0);
    for j = 1:length(lin_id)
        k = lin_id(j);
        x1 = 1 + mod(k-1, tr_len); x2 = 1 + floor((k-1) / tr_len);
        i1 = tr_ar(x1); i2 = tt_ar(x2);
        [mask, ptr, len] = append(mask, ptr, len, i1 + N * (i2 - 1));
    end
end
mask = mask(1:ptr-1);
end

function [hash_vals, idx_map, meta_map] = lsh(X, a, b, r)
hash_vals = floor((X * a' + b) ./ r);
idx_map = containers.Map;
meta_map = containers.Map;

for row=1:size(X,1)
    [idx_map, meta_map] = map_add(idx_map, meta_map, row, mat2str(hash_vals(row,:)));
end
end

function [idx_map, meta_map] = map_add(idx_map, meta_map, idx, hash_val)
%meta_map contains (hash_val -> [array_len, array_ptr]) key-value pairs
if isKey(meta_map, hash_val)
    entry = meta_map(hash_val);
    len = entry(1); ptr = entry(2);
    [new_ar, new_ptr, new_len] = append(idx_map(hash_val), ptr, len, idx);
    idx_map(hash_val) = new_ar;
    meta_map(hash_val) = [new_len, new_ptr];
else
    ar = zeros(1,200);
    ar(1) = idx;
    meta_map(hash_val) = [200, 2];
    idx_map(hash_val) = ar;
end
end

function [new_ar, new_ptr, new_len] = append(array, ptr, len, elm)
if ptr > len
    array = [array, zeros(1,200)];
    len = len + 200;
end
array(ptr) = elm;
new_ar = array; 
new_ptr = ptr + 1;
new_len = len;
end

function [a,b] = gen_ab(d, r, k)
a = mvnrnd(zeros(d,1), eye(d), k);
b = rand(1, k) * r;
end