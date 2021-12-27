% Standard normal variate
function spectra = snv(spectra, section, use_mad)

if exist('use_mad','var') == 1
    for i=1:size(spectra,1)
        if use_mad == 1
            spectra(i,:) = (spectra(i,:)-mean(spectra(i,:)))./mad(spectra(i,:));
        else
            spectra(i,:) = (spectra(i,:)-median(spectra(i,:)))./mad(spectra(i,:));
        end
    end
elseif exist('section','var') == 1
    for i=1:size(spectra,1)
        spectra(i,:) = (spectra(i,:)-mean(spectra(i,:)))./std(spectra(i,section));
    end
else
    for i=1:size(spectra,1)
        spectra(i,:) = (spectra(i,:)-mean(spectra(i,:)))./std(spectra(i,:));
    end
end
