function diamond_plot(SMI,C1,C2,Xlab,Ylab,ox,oy)
[n1,n2] = size(SMI);

% Prepare for arrows and equal signs
A_eq_B = zeros(n1,n2);
for i=1:n1
    for j=1:n2
        AB = all((1-10^-12) <= C1{i,j}(:,3) & C1{i,j}(:,2) <= (1+10^-12));
        BA = all((1-10^-12) <= C2{i,j}(:,3) & C2{i,j}(:,2) <= (1+10^-12));
        if AB && BA
            A_eq_B(i,j) = 2;
        elseif AB
            A_eq_B(i,j) = -1;
        elseif BA
            A_eq_B(i,j) = 1;
        else
            A_eq_B(i,j) = 0;
        end
    end
end

Cmap = colormap(gray(100)).^0.75;
hold on
for i=1:n1
    for j=1:n2
        x = -(i-j)/2; y = (i+j)/2;
        patch(x+[0 0.5 0 -0.5],y+[0 0.5 1 0.5],Cmap(max(1,round(100*SMI(i,j))),:));
        if A_eq_B(i,j) == 2
            text(x+0.0,y+0.5,'=','Color',[0,0,0],'HorizontalAlignment','center','FontSize',14)
        elseif A_eq_B(i,j) == 1
            text(x+0.05,y+0.55,'\supset','Color',[0.3,0.3,0.3],'HorizontalAlignment','center')
        elseif A_eq_B(i,j) == -1
            text(x-0.05,y+0.55,'\subset','Color',[0.3,0.3,0.3],'HorizontalAlignment','center')
        end
        
        if i==1
            text(x+0.5,y+0.0,num2str(j),'Color',[0,0,0],'HorizontalAlignment','center')
        end
        if j==1
            text(x-0.5,y+0.0,num2str(i),'Color',[0,0,0],'HorizontalAlignment','center')
        end
    end
end
if nargin > 5
    i = ox; j = oy;
    x = -(i-j)/2; y = (i+j)/2;
    p = patch(x+[0 0.5 0 -0.5],y+[0 0.5 1 0.5],Cmap(max(1,round(100*SMI(i,j))),:));
    if A_eq_B(i,j) == 2
        text(x+0.0,y+0.5,'=','Color',[0,0,0],'HorizontalAlignment','center','FontSize',14)
    elseif A_eq_B(i,j) == 1
        text(x+0.05,y+0.55,'\supset','Color',[0.3,0.3,0.3],'HorizontalAlignment','center')
    elseif A_eq_B(i,j) == -1
        text(x-0.05,y+0.55,'\subset','Color',[0.3,0.3,0.3],'HorizontalAlignment','center')
    end
    set(p,'EdgeColor','r','LineWidth',2)
end

xlim([-(n1+1)/2,(n2+1)/2])
ylim([0.5,(n1+n2+3)/2])
if nargin > 3
    text(-(n1+1)/2,(n1+n2+3)/8,Xlab,'FontSize',12)
    text((n2+1)/2,(n1+n2+3)/8,Ylab,'HorizontalAlignment','right','FontSize',12)
else
    text(-(n1+1)/2,(n1+n2+3)/8,'# comp. A','FontSize',12)
    text((n2+1)/2,(n1+n2+3)/8,'# comp. B','HorizontalAlignment','right','FontSize',12)
end
set(gca,'XTick',[])
set(gca,'YTick',[])

axis equal
colorbar
