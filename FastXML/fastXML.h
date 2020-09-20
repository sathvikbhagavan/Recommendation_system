#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <functional>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"

using namespace std;

extern LOGLVL loglvl;
extern mutex mtx;
extern _bool USE_IDCG;
extern thread_local mt19937 reng; // random number generator used during training 
extern thread_local VecF discounts;
extern thread_local VecF csum_discounts;
extern thread_local VecF dense_w;
extern thread_local VecI countmap;

class Param
{
public:
	_int num_Xf;
	_int num_Y;
	_float log_loss_coeff;
	_int max_leaf;
	_int lbl_per_leaf;
	_float bias;
	_int num_thread;
	_int start_tree;
	_int num_tree;
	_bool quiet;

	Param()
	{
		num_Xf = 0;
		num_Y = 0;
		log_loss_coeff = 1.0;
		max_leaf = 10;
		lbl_per_leaf = 100;
		bias = 1.0;
		num_thread = 1;
		start_tree = 0;
		num_tree = 50;
		quiet = false;
	}

	Param(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);
		fin >> (*this);
		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);
		fout << (*this); 
		fout.close();
	}

	friend istream& operator>>( istream& fin, Param& param )
	{
		fin >> param.num_Xf;
		fin >> param.num_Y;
		fin >> param.log_loss_coeff;
		fin >> param.max_leaf;
		fin >> param.lbl_per_leaf;
		fin >> param.bias;
		fin >> param.num_thread;
		fin >> param.start_tree;
		fin >> param.num_tree;
		fin >> param.quiet;
		return fin;
	}

	friend ostream& operator<<( ostream& fout, const Param& param )
	{
		fout << param.num_Xf << "\n";
		fout << param.num_Y << "\n";
		fout << param.log_loss_coeff << "\n";
		fout << param.max_leaf << "\n";
		fout << param.lbl_per_leaf << "\n";
		fout << param.bias << "\n";
		fout << param.num_thread << "\n";
		fout << param.start_tree<< "\n";
		fout << param.num_tree << "\n";
		fout << param.quiet << endl;
		return fout;
	}
};

class Node
{
public:
	_bool is_leaf;
	_int pos_child;
	_int neg_child;
	_int depth;
	VecI X;
	VecIF w;
	VecIF leaf_dist;

	Node()
	{
		is_leaf = false;
		depth = 0;
		pos_child = neg_child = -1;
	}

	Node(VecI X, _int depth, _int max_leaf)
	{
		this->X = X;
		this->depth = depth;
		this->pos_child = -1;
		this->neg_child = -1;

		if(X.size()<=max_leaf)
			this->is_leaf = true;
		else
			this->is_leaf = false;
	}

	~Node()
	{
	}

	_float get_ram()
	{
		_float ram = sizeof( Node );
		ram += X.size() * sizeof( _int );
		ram += w.size() * sizeof( pairIF );
		ram += leaf_dist.size() * sizeof( pairIF );
		return ram;
	}

	friend ostream& operator<<(ostream& fout, const Node& node)
	{
		fout<<(node.is_leaf?1:0)<<"\n";

		fout<<node.pos_child<<" "<<node.neg_child<<"\n";
		fout<<node.depth<<"\n";

		fout<<node.X.size();
		for(_int i=0; i<node.X.size(); i++)
			fout<<" "<<node.X[i];
		fout<<"\n";

		if(node.is_leaf)
		{
			fout<<node.leaf_dist.size();
			for(_int i=0; i<node.leaf_dist.size(); i++)
			{
				fout<<" "<<node.leaf_dist[i].first<<":"<<node.leaf_dist[i].second;
			}
			fout<<"\n";
		}
		else
		{
			fout<<node.w.size();
			for(_int i=0; i<node.w.size(); i++)
			{
				fout<<" "<<node.w[i].first<<":"<<node.w[i].second;
			}
			fout << "\n";
		}
		return fout;
	}

	friend istream& operator>>(istream& fin, Node& node)
	{
		fin>>node.is_leaf;
		fin>>node.pos_child>>node.neg_child>>node.depth;

		_int siz;
		_int ind;
		_float val;
		char c;

		node.X.clear();
		fin>>siz;
		for(_int i=0; i<siz; i++)
		{
			fin>>ind;	
			node.X.push_back(ind);
		}

		if(node.is_leaf)
		{
			node.leaf_dist.clear();
			fin>>siz;
			for(_int i=0; i<siz; i++)
			{
				fin>>ind>>c>>val;
				node.leaf_dist.push_back(make_pair(ind,val));
			}
		}
		else
		{
			node.w.clear();
			fin>>siz;
			for(_int i=0; i<siz; i++)
			{
				fin>>ind>>c>>val;
				node.w.push_back(make_pair(ind,val));
			}	
		}
		return fin;
	}
};

template <typename GNode>
class GTree // General instance tree with any kind of GNode. Supports FastXML, PfastreXML and SwiftXML
{
public:
	_int num_Xf;
	_int num_Y;
	vector<GNode*> nodes;

	GTree()
	{
		
	}

	GTree( string model_dir, _int tree_no )
	{
		ifstream fin;
		fin.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

		_int num_nodes;
		fin>>num_nodes;

		for(_int i=0; i<num_nodes; i++)
		{
			GNode* node = new GNode;
			fin>>(*node);
			nodes.push_back(node);
		}
		
		fin.close();
	}

	~GTree()
	{
		for(_int i=0; i<nodes.size(); i++)
			delete nodes[i];
	}

	_float get_ram()
	{
		_float ram = 0;
		for(_int i=0; i<nodes.size(); i++)
			ram += nodes[i]->get_ram();

		return ram;
	}

	void write( string model_dir, _int tree_no )
	{
		ofstream fout;
        fout.open( model_dir + "/" + to_string( tree_no ) + ".tree" );
		fout<<nodes.size()<<endl;

		for(_int i=0; i<nodes.size(); i++)
		{
			GNode* node = nodes[i];
			fout<<(*node);
		}

		fout.close();
	}
};

typedef GTree<Node> Tree;

Tree* train_tree(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, Param& param, _int tree_no);
void train_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, string model_dir, _float& train_time );


SMatF* predict_tree(SMatF* tst_ft_mat, Tree* tree, Param& param);
SMatF* predict_trees( SMatF* tst_X_Xf, Param& param, string model_dir, _float& prediction_time, _float& model_size );

_bool optimize_ndcg( SMatF* X_Y, VecI& pos_or_neg );
void calc_leaf_prob( Node* node, SMatF* X_Y, Param& param );
_bool optimize_log_loss( SMatF* Xf_X, VecI& y, VecF& C, VecIF& sparse_w, Param& param );
void setup_thread_locals( _int num_X, _int num_Xf, _int num_Y );
pairII get_pos_neg_count(VecI& pos_or_neg);
void test_svm( VecI& X, SMatF* X_Xf, VecIF& w, VecF& values );

