s = 0:0.25:1
[x,y,z] = ndgrid(s,s,s)
p = [x(:), y(:), z(:)]; %unperturbed points 
x1 = x; y1 = y; z1 = z;
x1(2:4,2:4,2:4) = x1(2:4,2:4,2:4) + 1e-2*randn(3,3,3); %perturb interior points
y1(2:4,2:4,2:4) = y1(2:4,2:4,2:4) + 1e-2*randn(3,3,3);
z1(2:4,2:4,2:4) = z1(2:4,2:4,2:4) + 1e-2*randn(3,3,3);
p1 = [x1(:), y1(:), z1(:)]; %perturbed points 
t = delaunayn(p);
t1 = delaunayn(p1);
[size(t),size(t1)]

% Forget about the original nodes p!

q = simpqual(p1,t);
q1 = simpqual(p1,t1);

[min(q),min(q1)]

simpplot(p1,t1, 'p(:,2)>0.5');
cameratoolbar

