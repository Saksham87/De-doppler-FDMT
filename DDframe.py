#Used to de-doppler a setigen frame

#Generation of setigen frame
frame = stg.Frame(fchans=1024*u.pixel,tchans=20*u.pixel,df=1*u.Hz,dt=10*u.s,fch1=6095*u.MHz)
noise = frame.add_noise(x_mean=10, noise_type='chi2')
signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=frame.fchans/2),
                                            drift_rate=-frame.df/frame.dt),
                                            stg.constant_t_profile(level=frame.get_intensity(snr=30)),
                                            stg.box_f_profile(width=1*u.Hz),
                                            stg.constant_bp_profile(level=1))

#Defining the de-doppler function
def ddframe(stgframe):
    
    fr=stgframe
    drift_rate=fr.df/fr.dt
    frequency = fr.dt*fr.tchans*drift_rate
    max_delay=int(frequency*2)

    d_cpu = np.expand_dims(fr.data, axis=0)
   
    ## Initialize FDMT
    n_disp = max_delay
    n_time = d_cpu.shape[2]
    n_chan = d_cpu.shape[1]
    fdmt.init(n_chan, n_disp, fr.fch1, fr.df, space="cuda", exponent=1)

    # Input shape is (1, n_freq, n_time)
    d_in = bf.ndarray(d_cpu, dtype='f32', space='cuda')
    d_out = bf.ndarray(np.zeros(shape=(1, n_disp, n_time)), dtype='f32', space='cuda')

    # Execute FDMT
    fdmt.execute(d_in, d_out, negative_delays=True)
    d_out = d_out.copy(space='system')
    
    d_in = bf.ndarray(d_cpu, dtype='f32', space='cuda')
    d_out2 = bf.ndarray(np.zeros(shape=(1, n_disp, n_time)), dtype='f32', space='cuda')
    
    fdmt.execute(d_in, d_out2, negative_delays=False)
    d_out2 = d_out2.copy(space='system')
    
    fal=np.array(d_out2)
    fal=np.flip(fal, axis=1)

    #plotting the fdmt
    plt.figure(figsize=(9, 6))
    plt.imshow(np.log(np.concatenate((fal,np.array(d_out)), axis=1).squeeze()), aspect='auto', extent=[0, fr.fchans, -((max_delay/fr.tchans)*(fr.df/fr.dt)), ((max_delay/fr.tchans)*(fr.df/fr.dt))])

    plt.xlabel("Frequency")
    plt.ylabel("De-doppler trial")
    plt.colorbar()
    
