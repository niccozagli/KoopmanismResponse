function [hnum,vnum] = decompose_genfn_numerical(x,y,obsnum,Kmax,W)

    [xgrid,ygrid] = meshgrid(x,y);

    kn = nchoosek(Kmax+2,Kmax);

    Inum = zeros(kn,1);

    index = 1;
    for i = 0:Kmax
        for j = 0:i
            
            dicnum = (xgrid.^j).*(ygrid.^(i-j));
            integrandnum = obsnum .* conj( dicnum );
            Inum(index) = trapz(y, trapz(x,integrandnum,2) );
            index = index + 1;
        end
    end
    
    PsiIntnum = zeros(kn);
    
    yindex = 1;
    for i = 0:Kmax
        for j = 0:i
            
            dicnum1 = (xgrid.^j).*(ygrid.^(i-j));
    
            xindex = 1; 
            
            for k = 0:Kmax
                for l = 0:k
                    
                    dicnum2 = (xgrid.^l).*(ygrid.^(k-l));
                    
                    integrand = dicnum2 .* conj( dicnum1 );
                    PsiIntnum(xindex,yindex) = trapz(y, trapz(x,integrand,2));
                    
                    xindex = xindex + 1;
    
                end
            end
            
            yindex = yindex + 1;
    
        end
    end
    
    hnum = inv(PsiIntnum.')*Inum;
    %hnum = Inum;
    
    vnum = W'*hnum;

end

