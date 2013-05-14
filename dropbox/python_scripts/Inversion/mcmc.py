def metro_hast(pdf,e_0,iternum = 500,sigma = 1.0):
    """ Implements a random-walk metropolis algorithm for a Markiv Chain process
        random walk implies the proposal distribution is symmetric (gaussian
        distributions suit this model)

        Inputs:
        pdf = a lambda function describing the PDF from which the draw is taken
        e_0 = an arbitrary starting point
        iternum = number of iterations (defaults to 500)
        sigma = the standard deviation of the gaussian random number generator (defaults to 1.0)

        Outputs:

        e_out = a list of length iternum giving the step-by-step reults of the random
                walk Markov chain.  A histogram of e_out should produce the correct distribution

        IMPORTANT NOTE:  This method requires a "burn-in period" during which the first 10% or so
        of results must be discarded """

    import numpy as np
    import random
    
    e_out = np.zeros(iternum)
    e_old = e_0

    for n in range(iternum):
        rand_norm = random.gauss(0,sigma)
        rand_uni = random.uniform(0,1)

        e_prime = e_old + rand_norm

        if pdf(e_prime) > pdf(e_old):
            e_out[n] = e_prime
        elif rand_uni <= (pdf(e_prime)/pdf(e_old)):
            e_out[n] = e_prime
        else:
            e_out[n] = e_old

        e_old = e_out[n]
        
    return e_out
            

if __name__ == '__main__':
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    alpha = 1e-5
    f = lambda x: np.exp(-alpha*x)
    
    x0 = 1
    
    posterior = np.zeros([10000,50])
    
    for n in range(10000):
        posterior[n,:] = metro_hast(f,x0,iternum=50, sigma=0.5)
    
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(2,1,1)
    ax1.hist(posterior[:,10],100)
    ax2 = fig1.add_subplot(2,1,2)
    ax2.plot(posterior[:,-1])
    fig1.canvas.draw()

    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(1,1,1)
    ax1.plot(np.mean(posterior, axis=0))
    fig2.canvas.draw()
    plt.show()

    

    
