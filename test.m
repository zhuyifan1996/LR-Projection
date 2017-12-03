function out = test
data = load('JSMF-nips/nips_N-5000_train.mat');
C = data.C;
% 
tic
X = tsvd(C);
toc

% X = randn(10000,20);
block_size = 1000;
% X = gen_fake(10000,20);
tic 
N = size(X,1);
if (N <= 1000) 
    C_train = X * X';
    i = C_train < 0;
    true_k = sum(sum(i));
else
    nBlock = floor(N / block_size);
    out = 0;
    for i = 1:nBlock
        for j = i+1:nBlock
            x1 = X((i-1)*block_size+1:i*block_size, :);
            x2 = X((j-1)*block_size+1:j*block_size, :);
            tmp = x1 * x2';
            out = out + sum(sum(tmp < 0));
        end
    end
    true_k = 2 * out;
end
fprintf("TRUTH: %i neg entries\n", true_k);
toc 

if 1
%     C = X*X';
    tic 
%     for i = 1:5
%         X = tsvd(C);
    mask = detect_neg(X, 1, 1);
    k = length(mask);
    fprintf('Found %i neg entries\n', k);
%         C(mask) = 0;
%     end
    toc
end
end

function fake_data = gen_fake(N, d)
fake_data = rand(N,d);
k = floor(N / 20);
neg_idx = randi(N, [1,k]);
fake_data(neg_idx, :) = randn(k, d);
end

function X = tsvd(C)
[U,S,V ] = svds(C, 20);
X = U * sqrt(S);
end

function [new_ar, new_ptr, new_len] = append(array, ptr, len, elm)
if ptr > len
    array = [array, zeros(1,10)];
    len = len + 10;
end
array(ptr) = elm;
new_ar = array; 
new_ptr = ptr + 1;
new_len = len;
end