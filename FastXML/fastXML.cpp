#include "fastXML.h"

using namespace std;

LOGLVL loglvl = LOGLVL::PROGRESS;  // print progress reports
mutex mtx;	// used to synchronize aggregation of individual tree scores during prediction time
_bool USE_IDCG = true; // if true, optimizes for nDCG; otherwise, optimized for DCG

/* reusable general data containers */ 
thread_local mt19937 reng; // random number generator used during training 
thread_local VecF discounts;
thread_local VecF csum_discounts;
thread_local VecF dense_w;
thread_local VecI countmap;

void setup_thread_locals( _int num_X, _int num_Xf, _int num_Y )
{
    discounts.resize( num_Y );
    csum_discounts.resize( num_Y+1 );
    
    csum_discounts[0] = 1.0;
    _float sumd = 0;
    for( _int i=0; i<num_Y; i++ )
    {
        discounts[i] = 1.0/log2((_float)(i+2));
        sumd += discounts[i];
        
        if(USE_IDCG)
            csum_discounts[i+1] = sumd;
		else
			csum_discounts[i+1] = 1.0;
    }
    dense_w.resize( num_Xf );
    for( _int i=0; i<num_Xf; i++ )
        dense_w[i] = 0;

	countmap.resize( max( num_Xf, num_Y ), 0 );
}

pairII get_pos_neg_count(VecI& pos_or_neg)
{
	pairII counts = make_pair(0,0);
	for(_int i=0; i<pos_or_neg.size(); i++)
	{
		if(pos_or_neg[i]==+1)
			counts.first++;
		else
			counts.second++;
	}
	return counts;
}

typedef signed char schar;
_bool optimize_log_loss( SMatF* Xf_X, VecI& y, VecF& C, VecIF& sparse_w, Param& param )
{
    _int num_X = Xf_X->nr;
    _int num_Xf = Xf_X->nc;
    _int* size = Xf_X->size;
    pairIF** data = Xf_X->data;
    
	_double eps = 0.01;    
	_int l = num_X;
	_int w_size = num_Xf;
	_int newton_iter=0, iter=0;
	_int max_newton_iter = 10;
	_int max_iter = 10;
	_int max_num_linesearch = 20;
	_int active_size;
	_int QP_active_size;

	_double nu = 1e-12;
	_double inner_eps = 1;
	_double sigma = 0.01;
	_double w_norm, w_norm_new;
	_double z, G, H;
	_double Gnorm1_init;
	_double Gmax_old = INF;
	_double Gmax_new, Gnorm1_new;
	_double QP_Gmax_old = INF;
	_double QP_Gmax_new, QP_Gnorm1_new;
	_double delta, negsum_xTd, cond;

    VecD w( num_Xf, 0 );
    VecI index( num_Xf, 0 );
    VecD Hdiag( num_Xf, 0 );
    VecD Grad( num_Xf, 0 );
    VecD wpd( num_Xf, 0 );
    VecD xjneg_sum( num_Xf, 0 );
    VecD xTd( num_X, 0 );
    VecD exp_wTx( num_X, 0 );
    VecD exp_wTx_new( num_X, 0 );
    VecD tau( num_X, 0 );
    VecD D( num_X, 0 );
	
	w_norm = 0;
	for( _int i=0; i<w_size; i++ )
	{
		index[i] = i;

        for( _int j=0; j<size[i]; j++ )
		{
			_int inst = data[i][j].first;
			_float val = data[i][j].second;

			if(y[inst] == -1)
				xjneg_sum[i] += C[inst]*val;
		}
	}

	for( _int i=0; i<l; i++ )
	{
		exp_wTx[i] = exp(exp_wTx[i]);
		_double tau_tmp = 1/(1+exp_wTx[i]);
		tau[i] = C[i]*tau_tmp;
		D[i] = C[i]*exp_wTx[i]*SQ(tau_tmp);
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(_int s=0; s<active_size; s++)
		{
			_int i = index[s];
			Hdiag[i] = nu;

			_double tmp = 0;
		
			for( _int j=0; j<size[i]; j++ )
			{
				_int inst = data[i][j].first;
				_float val = data[i][j].second;
				Hdiag[i] += SQ(val)*D[inst];
				tmp += val*tau[inst];
			}

			Grad[i] = -tmp + xjneg_sum[i];

			_double Gp = Grad[i]+1;
			_double Gn = Grad[i]-1;
			_double violation = 0;

			if(w[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(_int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(_int i=0; i<QP_active_size; i++)
			{
				_llint r = reng();
				_int j = i+r%(QP_active_size-i);
				swap(index[j], index[i]);
			}

			for(_int s=0; s<QP_active_size; s++)
			{
				_int i = index[s];
				H = Hdiag[i];

				G = Grad[i] + (wpd[i]-w[i])*nu;
				for( _int j=0; j<size[i]; j++ )
				{
					_int inst = data[i][j].first;
					_float val = data[i][j].second;
					G += val*D[inst]*xTd[inst];
				}

				_double Gp = G+1;
				_double Gn = G-1;
				_double violation = 0;
				if(wpd[i] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[i] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[i])
					z = -Gp/H;
				else if(Gn > H*wpd[i])
					z = -Gn/H;
				else
					z = -wpd[i];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[i] += z;

				for( _int j=0; j<size[i]; j++ )
				{
					_int inst = data[i][j].first;
					_float val = data[i][j].second;
					xTd[inst] += val*z;
				}
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		delta = 0;
		w_norm_new = 0;
		for(_int i=0; i<w_size; i++)
		{
			delta += Grad[i]*(wpd[i]-w[i]);
			if(wpd[i] != 0)
				w_norm_new += fabs(wpd[i]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(_int i=0; i<l; i++)
		{
			if(y[i] == -1)
				negsum_xTd += C[i]*xTd[i];
		}

		_int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			_double cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(_int i=0; i<l; i++)
			{
				_double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[i]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(_int i=0; i<w_size; i++)
					w[i] = wpd[i];

				for(_int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					_double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[i]*tau_tmp;
					D[i] = C[i]*exp_wTx[i]*SQ(tau_tmp);
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(_int i=0; i<w_size; i++)
				{
					wpd[i] = (w[i]+wpd[i])*0.5;

					if(wpd[i] != 0)
						w_norm_new += fabs(wpd[i]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(_int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(_int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(_int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;

				for( _int j=0; j<size[i]; j++ )
				{
					_int inst = data[i][j].first;
					_float val = data[i][j].second;
					exp_wTx[inst] += w[i]*val;
				}
			}

			for(_int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;
	}

	_float th = 1e-16;
	for( _int i=0; i<w_size; i++ )
	{
		if(fabs(w[i])>th)
			sparse_w.push_back(make_pair(i,w[i]));
        else
            w[i] = 0;
	}
    
    VecF prods( l, 0 );
    for( _int i=0; i<w_size; i++ )
	{
        for( _int j=0; j<size[i]; j++ )
		{
			_int inst = data[i][j].first;
			_float val = data[i][j].second;
            prods[inst] += w[i]*val;
		}
	}
    
    for( _int i=0; i<l; i++ )
        y[i] = prods[i]>=0 ? +1 : -1;
    
	pairII num_pos_neg = get_pos_neg_count(y);

	if(num_pos_neg.first==0 || num_pos_neg.second==0)
	{
		sparse_w.clear();
		return false;
	}

	return true;
}

void calc_leaf_prob( Node* node, SMatF* X_Y, Param& param )
{
    _int lbl_per_leaf = param.lbl_per_leaf;
    _int num_X = X_Y->nc;
    _int num_Y = X_Y->nr;
    _int* size = X_Y->size;
    pairIF** data = X_Y->data;
    
	VecIF& leaf_dist = node->leaf_dist;
	leaf_dist.resize( num_Y );
	for( _int i=0; i<num_Y; i++ )
		leaf_dist[i] = make_pair( i, 0 );

	for( _int i=0; i<num_X; i++ )
	{
		for( _int j=0; j<size[i]; j++ )
        {
			_int lbl = data[i][j].first;
			_float val = data[i][j].second;
			leaf_dist[lbl].second += val;
		}
	}	

	for( _int i=0; i<num_Y; i++ )
		leaf_dist[i].second /= num_X;

	sort( leaf_dist.begin(), leaf_dist.end(), comp_pair_by_second_desc<_int,_float> );
	if( leaf_dist.size()>lbl_per_leaf )
		leaf_dist.resize( lbl_per_leaf );
	sort( leaf_dist.begin(), leaf_dist.end(), comp_pair_by_first<_int,_float> );
}

_bool optimize_ndcg( SMatF* X_Y, VecI& pos_or_neg )
{
    _int num_X = X_Y->nc;
    _int num_Y = X_Y->nr;
    _int* size = X_Y->size;
    pairIF** data = X_Y->data;
    
    _float eps = 1e-6;
    
    VecF idcgs( num_X );
    for( _int i=0; i<num_X; i++ )
        idcgs[i] = 1.0/csum_discounts[ size[i] ];
    
    VecIF pos_sum( num_Y );
    VecIF neg_sum( num_Y );
    VecF diff_vec( num_Y );

    _float ndcg = -2;
    _float new_ndcg = -1;
    
	while(true)
	{
		for(_int i=0; i<num_Y; i++ )
		{
			pos_sum[i] = make_pair(i,0);
			neg_sum[i] = make_pair(i,0);
			diff_vec[i] = 0;
		}

		for( _int i=0; i<num_X; i++ )
		{
			for( _int j=0; j<size[i]; j++ )
			{
				_int lbl = data[i][j].first;
				_float val = data[i][j].second * idcgs[i];

				if(pos_or_neg[i]==+1)
					pos_sum[lbl].second += val;
				else
					neg_sum[lbl].second += val;
			}
		}

		new_ndcg = 0;
		for(_int s=-1; s<=1; s+=2)
		{
			VecIF& sum = s==-1 ? neg_sum : pos_sum;
			sort(sum.begin(), sum.begin()+num_Y, comp_pair_by_second_desc<_int,_float>);

			for(_int i=0; i<num_Y; i++)
			{
				_int lbl = sum[i].first;
				_float val = sum[i].second;
				diff_vec[lbl] += s*discounts[i];
				new_ndcg += discounts[i]*val;
			}
		}
		new_ndcg /= num_X;

		for( _int i=0; i<num_X; i++ )
		{
			_float gain_diff = 0;
            for( _int j=0; j<size[i]; j++ )
			{
                _int lbl = data[i][j].first;
				_float val = data[i][j].second * idcgs[i];
				gain_diff += val*diff_vec[lbl];
			}

			if(gain_diff>0)
				pos_or_neg[i] = +1;
			else if(gain_diff<0)
				pos_or_neg[i] = -1;
		}
	
		if(new_ndcg-ndcg<eps)
			break;
		else
			ndcg = new_ndcg;

	}

	pairII num_pos_neg = get_pos_neg_count(pos_or_neg);
	if(num_pos_neg.first==0 || num_pos_neg.second==0)
		return false;
	return true;
}

void shrink_data_matrices( SMatF* trn_X_Xf, SMatF* trn_X_Y, VecI& n_X, SMatF*& n_trn_Xf_X, SMatF*& n_trn_X_Y, VecI& n_Xf, VecI& n_Y )
{
    trn_X_Xf->shrink_mat( n_X, n_trn_Xf_X, n_Xf, countmap, true ); // countmap is a thread_local variable
    trn_X_Y->shrink_mat( n_X, n_trn_X_Y, n_Y, countmap, false );
}

_bool split_node( Node* node, SMatF* Xf_X, SMatF* X_Y, VecI& pos_or_neg, Param& param )
{
    _int num_X = Xf_X->nr;
	pos_or_neg.resize( num_X );
 
	for( _int i=0; i<num_X; i++ )
	{
		_llint r = reng();

		if(r%2)
			pos_or_neg[i] = 1;
		else
			pos_or_neg[i] = -1;
	}

	// one run of ndcg optimization
	bool success;

	success = optimize_ndcg( X_Y, pos_or_neg );
	if(!success)
		return false;

	VecF C( num_X );
	pairII num_pos_neg = get_pos_neg_count( pos_or_neg );
	_float frac_pos = (_float)num_pos_neg.first/(num_pos_neg.first+num_pos_neg.second);
	_float frac_neg = (_float)num_pos_neg.second/(num_pos_neg.first+num_pos_neg.second);
	_double Cp = param.log_loss_coeff/frac_pos;
	_double Cn = param.log_loss_coeff/frac_neg;  // unequal Cp,Cn improves the balancing in some data sets
	
	for( _int i=0; i<num_X; i++ )
		C[i] = pos_or_neg[i]==+1 ? Cp : Cn;

	// one run of log-loss optimization
	success = optimize_log_loss( Xf_X, pos_or_neg, C, node->w, param );
	if(!success)
		return false;

	return true;
}

void postprocess_node( Node* node, SMatF* trn_X_Xf, SMatF* trn_X_Y, VecI& n_X, VecI& n_Xf, VecI& n_Y )
{
    if( node->is_leaf )
        reindex_VecIF( node->leaf_dist, n_Y );
    else
        reindex_VecIF( node->w, n_Xf );
}

Tree* train_tree( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, _int tree_no )
{
	reng.seed(tree_no);

	_int num_X = trn_X_Xf->nc;
	_int num_Xf = trn_X_Xf->nr;
	_int num_Y = trn_X_Y->nr;

	Tree* tree = new Tree;
	vector<Node*>& nodes = tree->nodes;

	VecI X;
	for(_int i=0; i<num_X; i++)
		X.push_back(i);
	Node* root = new Node( X, 0, param.max_leaf );
	nodes.push_back(root);

	VecI pos_or_neg;

	for(_int i=0; i<nodes.size(); i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
		{
			if(i%1000==0)
				cout<<"\tnode "<<i<<endl;
		}		

		Node* node = nodes[i];
		VecI& n_X = node->X;	
		SMatF* n_trn_Xf_X;
		SMatF* n_trn_X_Y;
		VecI n_Xf;
        VecI n_Y;

		shrink_data_matrices( trn_X_Xf, trn_X_Y, n_X, n_trn_Xf_X, n_trn_X_Y, n_Xf, n_Y );

		if(node->is_leaf)
		{
			calc_leaf_prob( node, n_trn_X_Y, param );
        }
		else
		{
			VecI pos_or_neg;
			bool success = split_node( node, n_trn_Xf_X, n_trn_X_Y, pos_or_neg, param );

			if(success)
			{
				VecI pos_X, neg_X;
				for(_int j=0; j<n_X.size(); j++)
				{
					_int inst = n_X[j];
					if( pos_or_neg[j]==+1 )
						pos_X.push_back(inst);
					else
						neg_X.push_back(inst);
				}
	
				Node* pos_node = new Node( pos_X, node->depth+1, param.max_leaf );
				nodes.push_back(pos_node);
				node->pos_child = nodes.size()-1;

				Node* neg_node = new Node( neg_X, node->depth+1, param.max_leaf );
				nodes.push_back(neg_node);
				node->neg_child = nodes.size()-1;
			}
			else
			{
				node->is_leaf = true;
				i--;
			}
		}
       
        postprocess_node( node, trn_X_Xf, trn_X_Y, n_X, n_Xf, n_Y );

		delete n_trn_Xf_X;
		delete n_trn_X_Y;
	}
	tree->num_Xf = num_Xf;
	tree->num_Y = num_Y;

	return tree;
}

void train_trees_thread( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param param, _int s, _int t, string model_dir, _float* train_time )
{
	Timer timer;
	timer.tic();
    _int num_X = trn_X_Xf->nc;
    _int num_Xf = trn_X_Xf->nr;
    _int num_Y = trn_X_Y->nr;
    setup_thread_locals( num_X, num_Xf, num_Y );
    {
		lock_guard<mutex> lock(mtx);
		*train_time += timer.toc();
    }
    
	for(_int i=s; i<s+t; i++)
	{
		timer.tic();
		cout<<"tree "<<i<<" training started"<<endl;

		Tree* tree = train_tree( trn_X_Xf, trn_X_Y, param, i );
		{
			lock_guard<mutex> lock(mtx);
			*train_time += timer.toc();
		}

		tree->write( model_dir, i );

		timer.tic();
		delete tree;

		cout<<"tree "<<i<<" training completed"<<endl;
		
		{
			lock_guard<mutex> lock(mtx);
			*train_time += timer.toc();
		}
	}
}

void train_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, string model_dir, _float& train_time )
{
	_float* t_time = new _float;
	*t_time = 0;
	Timer timer;
	
	timer.tic();
	trn_X_Xf->append_bias_feat( param.bias );

	_int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
	vector<thread> threads;
	_int s = param.start_tree;
	for( _int i=0; i<param.num_thread; i++ )
	{
		if( s < param.start_tree+param.num_tree )
		{
			_int t = min( tree_per_thread, param.start_tree+param.num_tree-s );
			threads.push_back( thread( train_trees_thread, trn_X_Xf, trn_X_Y, param, s, t, model_dir, ref(t_time) ));
			s += t;
		}
	}
	*t_time += timer.toc();	

	for(_int i=0; i<threads.size(); i++)
		threads[i].join();

	train_time = *t_time;
	delete t_time;
}

void test_svm( VecI& X, SMatF* X_Xf, VecIF& w, VecF& values )
{
	values.resize( X.size() );
    _int num_Xf = X_Xf->nr;

	for(_int i=0; i<w.size(); i++)
		dense_w[w[i].first] = w[i].second;

	_int* siz = X_Xf->size;
	pairIF** data = X_Xf->data;

	for(_int i=0; i<X.size(); i++)
	{
		_int inst = X[i];
		_float prod = 0;

		for(_int j=0; j<siz[inst]; j++)
		{
			_int ft = data[inst][j].first;
			_float val = data[inst][j].second;
			prod += val*dense_w[ft];
		}

		values[i] = prod;
	}	

	for(_int i=0; i<w.size(); i++)
		dense_w[w[i].first] = 0;
}

SMatF* predict_tree( SMatF* tst_X_Xf, Tree* tree, Param& param )
{
	_int num_X = tst_X_Xf->nc;
	_int num_Xf = param.num_Xf;
	_int num_Y = param.num_Y;

	vector<Node*>& nodes = tree->nodes;
	Node* node = nodes[0];
	node->X.clear();

	for(_int i=0; i<num_X; i++)
		node->X.push_back(i);

	SMatF* tst_score_mat = new SMatF(num_Y,num_X);
	VecI pos_or_neg( num_X );
	VecF values( num_X );

	for(_int i=0; i<nodes.size(); i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
		{
			if(i%1000==0)
				cout<<"\tnode "<<i<<endl;
		}		

		Node* node = nodes[i];
	
		if(!node->is_leaf)
		{
			VecI& X = node->X;
			test_svm(X, tst_X_Xf, node->w, values);
			for( _int j=0; j<X.size(); j++ )
				pos_or_neg[j] = values[j]>=0 ? +1 : -1;
			Node* pos_node = nodes[node->pos_child];
			pos_node->X.clear();
			Node* neg_node = nodes[node->neg_child];
			neg_node->X.clear();

			for(_int j=0; j<X.size(); j++)
			{
				if(pos_or_neg[j]==+1)
					pos_node->X.push_back(X[j]);
				else
					neg_node->X.push_back(X[j]);
			}
		}
		else
		{
			VecI& X = node->X;
			VecIF& leaf_dist = node->leaf_dist;
			_int* size = tst_score_mat->size;
			pairIF** data = tst_score_mat->data;

			for(_int j=0; j<X.size(); j++)
			{
				_int inst = X[j];
				size[inst] = leaf_dist.size();
				data[inst] = new pairIF[leaf_dist.size()];

				for(_int k=0; k<leaf_dist.size(); k++)
					data[inst][k] = leaf_dist[k];
			}
		}
	}

	return tst_score_mat;
}

void predict_trees_thread( SMatF* tst_X_Xf, SMatF* score_mat, Param param, _int s, _int t, string model_dir, _float* prediction_time, _float* model_size )
{
    Timer timer;
    
    timer.tic();
    _int num_Xf = tst_X_Xf->nr;
    dense_w.resize( num_Xf );
    for( _int i=0; i<num_Xf; i++ )
        dense_w[i] = 0;
	{
		lock_guard<mutex> lock(mtx);
		*prediction_time += timer.toc();
	}

	for(_int i=s; i<s+t; i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" prediction started"<<endl;

		Tree* tree = new Tree( model_dir, i );
        timer.tic();
		SMatF* tree_score_mat = predict_tree( tst_X_Xf, tree, param );

		{
			lock_guard<mutex> lock(mtx);
			score_mat->add(tree_score_mat);
            *model_size += tree->get_ram();
		}

		delete tree;
		delete tree_score_mat;

		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" prediction completed"<<endl;
        {
			lock_guard<mutex> lock(mtx);
			*prediction_time += timer.toc();
		}
	}
}

SMatF* predict_trees( SMatF* tst_X_Xf, Param& param, string model_dir, _float& prediction_time, _float& model_size )
{
    _float* p_time = new _float;
	*p_time = 0;

	_float* m_size = new _float;
	*m_size = 0;

	Timer timer;

	timer.tic();
    tst_X_Xf->append_bias_feat( param.bias );
    
    _int num_X = tst_X_Xf->nc;
    SMatF* score_mat = new SMatF( param.num_Y, num_X );

	_int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
	vector<thread> threads;

	_int s = param.start_tree;
	for(_int i=0; i<param.num_thread; i++)
	{
		if(s < param.start_tree+param.num_tree)
		{
			_int t = min(tree_per_thread, param.start_tree+param.num_tree-s);
            threads.push_back( thread( predict_trees_thread, tst_X_Xf, ref(score_mat), param, s, t, model_dir, ref( p_time ), ref( m_size ) ));
			s += t;
		}
	}
    *p_time += timer.toc();
	
	for(_int i=0; i<threads.size(); i++)
		threads[i].join();

    timer.tic();

	for(_int i=0; i<score_mat->nc; i++)
		for(_int j=0; j<score_mat->size[i]; j++)
			score_mat->data[i][j].second /= param.num_tree;
    
    model_size = *m_size;
	delete m_size;
    
    *p_time += timer.toc();
	prediction_time = *p_time;
	delete p_time;

	return score_mat;
}
