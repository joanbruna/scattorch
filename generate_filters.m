clear all

addpath('~/matlab/toolbox/synthesis/rotosynthesis')
addpath('~/matlab/scatnet')

cd('~/projects/scattorch')

for c=1:3
for j=1:4
generate_scattorch_filters(c,j);
end
end


