function [R,P,T,Q,U,B,Xmeans,Ymeans,fitted,residuals,Xvar,Xtotvar] = simpls(ncomp, X, Y)
[nobj,npred]  = size(X);
nresp = size(Y,2);
V  = zeros(npred, ncomp);
R  = zeros(npred, ncomp);
tQ = zeros(ncomp, nresp);
B  = zeros(npred, nresp, ncomp);
P  = R;
U  = zeros(nobj, ncomp);
T  = zeros(nobj, ncomp);
fitted = zeros(nobj, nresp, ncomp);
Xmeans = mean(X);
Ymeans = mean(Y);
X = bsxfun(@minus, X, Xmeans);
Y = bsxfun(@minus, Y, Ymeans);
S = X'*Y;
for a = 1:ncomp
    if nresp == 1
        q_a = 1;
    else
        if nresp < npred
            [q_a,eq_a] = eig(S'*S);
            [~,m] = max(abs(diag(eq_a)));
            q_a = q_a(:,m);
        else
            [s,es_a] = eig(S*S');
            [~,m] = max(abs(diag(es_a)));
            s = s(:,m);
            q_a = S'*s;
            q_a = q_a./sqrt(q_a'*q_a);
        end
    end
    r_a = S * q_a;
    t_a = X * r_a;
    t_a = t_a - mean(t_a);
    tnorm = sqrt(t_a'*t_a);
    t_a = t_a./tnorm;
    r_a = r_a./tnorm;
    p_a = X'*t_a;
    q_a = Y'*t_a;
    v_a = p_a;
    if (a > 1)
        v_a = v_a - V * (V' * p_a);
    end
    v_a = v_a./sqrt(v_a'*v_a);
    S = S - v_a * (v_a' * S);
    R(:,a)  = r_a;
%     [W,~] = qr(R(:,1:a),0);
    tQ(a,:) = q_a;
    V(:,a)  = v_a;
    if nargout > 5
        B(:,:,a) = R(:,1:a) * tQ(1:a,:);
    end
    u_a = Y * q_a;
    u_a = u_a - T * (T' * u_a);
    P(:,a) = p_a;
    T(:,a) = t_a;
    U(:,a) = u_a;
    if nargout > 9
        fitted(:,:,a) = T(:,1:a) * tQ(1:a,:);
    end
end
Q = tQ';
if nargout > 8
    residuals = -fitted + repmat(Y, [1, nresp, ncomp]);
end
if nargout > 9
    fitted = fitted + repmat(Ymeans, [nobj, 1, ncomp]);
end
if nargout > 10
    Xvar = sum(P.*P);
    Xtotvar = sum(sum(X.*X));
end
