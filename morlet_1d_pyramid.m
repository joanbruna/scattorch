function filters = morlet_1d_pyramid(options)

%sigma=.8;%2/sqrt(3); 
%sigma0=.8;%2*sigma/3;
%sigma1 = sqrt(2*sigma^2-sigma0^2); is sigmap
%sigma2 = sqrt(1.5)*sigma0; is sigma
%xi = 2*pi/3;
%xi1 = xi*3/4;% * 2 * sigma^2/sigma1^2

Q = getoptions(options,'Q',1);
N = getoptions(options,'length',9*Q);
Ndowns = getoptions(options,'lengthdowns',9);
sigmap0 = getoptions(options,'sigmap0', 0.25*sqrt(Q));
sigma0 = getoptions(options,'sigma0', 0.25);
sigmap = getoptions(options, 'sigmap',0.25*sqrt(Q) );
sigma = getoptions(options, 'sigma', sigma0);
xi = getoptions(options,'xi',0.5*pi);
xi0 = getoptions(options,'xi0',xi*3/4);

%xi = getoptions(options,'xi',2*pi/3);
%xi0 = getoptions(options,'xi0',xi*3/4);

freqspan = linspace(-0.5,0.5,2+Q);
freqspan = freqspan(2:end-1);
freqspan = 2.^freqspan;



offset = floor(N/2);
x = [1:N] - offset -1 ;
x =x';

filters.identity = zeros(N,1);
filters.identity(ceil(size(filters.identity,1)/2))=1/sqrt(2);

filters.h0 = gausswin(N, 1/sigma0);
filters.h = gausswin(N, 1/sigma);

for q=1:Q
    filters.g0{q} = gausswin(N, (freqspan(q)/sigmap0));
    oscilating_part = filters.g0{q} .* exp(1i*x*xi0*freqspan(q));
    K = sum(oscilating_part(:)) ./ sum(filters.g0{q}(:));
    filters.g0{q} = oscilating_part - K.*filters.g0{q};
    
    filters.g{q} = gausswin(N, (freqspan(q)/sigmap));
    oscilating_part = filters.g{q} .* exp(1i*x*xi*freqspan(q));
    K = sum(oscilating_part(:)) ./ sum(filters.g{q}(:));
    filters.g{q} = oscilating_part - K.*filters.g{q};

end

filters.downfilters = gausswin(Ndowns, 1/sigma);

%renormalize
fact = getoptions(options,'renfact',sqrt(1));

filters.h0 = filters.h0 * fact / sum(abs(filters.h0));
filters.h = filters.h * fact / sum(abs(filters.h));
for q=1:Q
filters.g0{q} = filters.g0{q} * fact / sum(abs(filters.g0{q}));
filters.g{q} = filters.g{q} * fact / sum(abs(filters.g{q}));
end
filters.downfilters = filters.downfilters * sqrt(2) / sum(abs(filters.downfilters));

