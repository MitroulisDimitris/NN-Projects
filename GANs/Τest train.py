#%% Τest train 
batch_sizes = [50,100,200,500,750,1000,2000,5000]


img_list = []
G_losses = []
D_losses = []
iters = 0

#for i, (imgs, labels) in enumerate(train_loader):

if __name__ == '__main__':
    for epoch in range(len(batch_sizes)):
        yes_loader = get_loader(batch_sizes[epoch])
        st = time.time()
        for i, (imgs, labels) in enumerate(yes_loader):
            tr_g, tr_d = False, False   
            # -----------------
            # Things to use later
            # -----------------
            imgs = imgs.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM))))
    
            # transform to tensor [256,1,28,28]
            real_imgs = Variable(imgs.type(Tensor))
            fake_labels = build_fake_labels(labels.to(device))
    
            labels = labels.to(device)
    
            # -----------------
            # Train discriminator
            # -----------------
    
            #Forward pass through Discriminator
            real_pred = discriminator(real_imgs,labels)
            real_loss = loss_func(real_pred, valid)
            #real_loss.backward()
            #D_x = real_pred.mean().item()
    
            # Generate a batch of images
            gen_imgs = generator(z,labels)
            
            # Calculate D's loss on the all-fake batch
            fake_pred = discriminator(gen_imgs.detach(),labels.detach())
            fake_loss = loss_func(fake_pred, fake)
            #fake_loss.backward()
            #D_G_z1 = fake_pred.mean().item()
            #d_loss = fake_pred + real_pred
    
           
    
            
            # -----------------
            #  Train Generator
            # -----------------
            
            #Loss measures generator's ability to fool the discriminator
            #fake_pred = discriminator(gen_imgs,labels)
            #g_loss = loss_func(fake_pred, valid)
            #g_loss.backward()
            #D_G_z2 = fake_pred.mean().item()
           
          
            
                
            tr_d = True
            optimizer_D.zero_grad()
            real_loss.backward()
            fake_loss.backward()
            D_x = real_pred.mean().item()
            D_G_z1 = fake_pred.mean().item()
            d_loss = (fake_pred + real_pred)/2
            
            
            optimizer_D.step()
              
    
            
            tr_g = True
            optimizer_G.zero_grad()
            
            fake_pred = discriminator(gen_imgs,labels)
            g_loss = loss_func(fake_pred, valid)
            g_loss.backward()
            D_G_z2 = fake_pred.mean().item()
            optimizer_G.step()
               
            
             
    
          
    
            
           # Output training stats
            #if i % 50 == 0:
                #print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f tD(x): %.4f tD(G(z)): %.4f / %.4f,train D,G = %s,%s'
                      #% (epoch, NUM_EPOCHS, i, len(train_loader),d_loss.mean().item(), g_loss.item(), D_x, D_G_z1, D_G_z2,tr_d,tr_g))
    
            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(d_loss[0].item())
    
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = generator(z,labels).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    
            iters += 1
            
        end = time.time()-st
        print('time: %.4f , batch_size = %d' % (end,batch_sizes[epoch]))
        
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    