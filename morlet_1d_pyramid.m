function filters = morlet_1d_pyramid(options)

%sigma=.8;%2/sqrt(3); 
%sigma0=.8;%2*sigma/3;
%sigma1 = sqrt(2*sigma^2-sigma0^2); is sigmap
%sigma2 = sqrt(1.5)*sigma0; is sigma
%xi = 2*pi/3;
%xi1 = xi*3/4;% * 2 * sigma^2/sigma1^2

N = getoptions(options,'length',9);
Ndowns = getoptions(options,'lengthdowns',5);
sigmap0 = getoptions(options,'sigmap0', 0.6);
sigma0 = getoptions(options,'sigma0', 0.6);
sigmap = getoptions(options, 'sigmap',sqrt(2*sigmap0^2-sigma0^2) );
sigma = getoptions(options, 'sigma', sqrt(1.0)*sigma0);
xi = getoptions(options,'xi',2*pi/3);
xi0 = getoptions(options,'xi0',xi*3/4);

offset = floor(N/2);
x = [1:N] - offset -1 ;
x =x';

filters.identity = zeros(N,1);
filters.identity(ceil(size(filters.identity,1)/2))=1;

filters.h0 = gausswin(N, 1/sigma0);
filters.h = gausswin(N, 1/sigma);

filters.g0 = gausswin(N, 1/sigmap0);
oscilating_part = filters.g0 .* exp(1i*x*xi0);
K = sum(oscilating_part(:)) ./ sum(filters.g0(:));
filters.g0 = oscilating_part - K.*filters.g0;

filters.g = gausswin(N, 1/sigmap);
oscilating_part = filters.g .* exp(1i*x*xi);
K = sum(oscilating_part(:)) ./ sum(filters.g(:));
filters.g = oscilating_part - K.*filters.g;

filters.downfilters = gausswin(Ndowns, 1/sigma);
