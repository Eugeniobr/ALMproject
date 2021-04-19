#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <ilcplex/ilocplex.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <chrono>
using namespace std::chrono;

using namespace std;
ILOSTLBEGIN


int main (int argc, char *argv[]){




    try{

    ifstream arq(argv[1]);

    omp_set_num_threads(stoi(argv[2]));

    ofstream output1, output2;




    if (!arq) {cerr << "Erro arquivo \n"; exit(0);}


	int N; // N ativos
    int T; // T estágios
    int branching; //ramificacao
    int S; // S cenarios
    int mu;
    int sb_size; // tamanho do subproblema
    int sb_stages;
    int inst;
    float M; // constante do modelo
    float Q; // recursos iniciais
    float K; // threshold do funding ratio
    float P_inicial;
    double rho;



    arq >> inst;
    arq >> N;
    arq >> T;
    arq >> branching;
    arq >> M;
    arq >> Q;
    arq >> K;
    vector<float> pi(N); //peso maximo dos ativos
    arq >> pi[0];
    arq >> pi[1];
    arq >> P_inicial;
    S = pow(branching, T);
    stringstream s1, s2, s3;
    s1 << branching;
    s2 << T;
    s3 << inst;
    string output2_name = s1.str();
    string name2 = s2.str();
    string name3 = s3.str();
    string output1_name = "resultados";
    string inst_name;
    output2_name = output2_name.append("-");
    output2_name = output2_name.append(name2);
    output2_name = output2_name.append("-");
    output2_name = output2_name.append(name3);
    inst_name = output2_name;
    output2_name = output2_name.append("-3");
    output1_name = output1_name.append(inst_name);
    output2.open(output2_name, ofstream::app);

    if(branching == 2 || branching == 3)
        sb_stages = 7;
    else
        sb_stages = 5;

    sb_size = pow(branching, sb_stages);
    output2 << sb_size << endl;
    output1.open(output1_name, ofstream::app);

    vector<float> lt(T); // valor dos passivos no tempo t
    vector<float> ft(T); // valor das contribuiçoes no tempo t
    vector<float> Lt(T); // valor presente dos passivos do tempo t, t+1, ..., T
    vector<float> Ft(T); // valor presente das contruições do tempo t, t+1, ...., T
    vector<vector<vector<double> > > P(N, vector<vector<double> >(T+1, vector<double>(S))); //matriz de precos
    //vector<vector<vector<double>>> X1(N, vector<vector<double>>(T+1, vector<double>(S)));
    //vector<vector<vector<double>>> X_at(N, vector<vector<double>>(T+1, vector<double>(S)));
    //vector<vector<vector<double>>> lambda1(N, vector<vector<double>>(T+1, vector<double>(S)));
    vector<vector<vector<float> > > fix_count(T, vector<vector<float> >(S, vector<float>(5)));
    vector<vector<float> > C1(T, vector<float>(S));



    /*

	cout << N << endl;
	cout << T << endl;
	cout << branching << endl;
	cout << M << endl;
	cout << Q << endl;
	cout << K << endl;
	cout << pi[0] << endl;
	cout << pi[1] << endl;
    */

    for(int i=0; i < N; i++){


        for(int k=0; k < S; k++){

           for(int j=0; j <= T; j++){

                if(j == 0)
                    P[i][j][k] = P_inicial;

                else{
                    arq >> P[i][j][k];
                }

            }

        }

    }



    for(int j=0; j< T; j++)
        arq >> lt[j];

    for(int j=0; j< T; j++)
        arq >> Lt[j];

    for(int j=0; j< T; j++)
        arq >> ft[j];

    for(int j=0; j< T; j++)
        arq >> Ft[j];







        for(int nr = 1; nr <= 4; nr++){


            int last_ite;
            float last_opt;
            rho = 0.01/pow(10, nr);
            vector<vector<vector<float> > > lambda1(N, vector<vector<float> >(T+1, vector<float>(S)));
            vector<vector<vector<float> > > X1(N, vector<vector<float> >(T+1, vector<float>(S)));
            vector<vector<vector<float> > > X_at(N, vector<vector<float> >(T+1, vector<float>(S)));
            vector<vector<vector<float> > > X_aux(N, vector<vector<float> >(T+1, vector<float>(S)));
            output1 << inst_name << "\t" << "3" << "\t" << rho;
            output2 << "#" << endl;
            output2 << rho << endl;
            auto start = high_resolution_clock::now();
            int cont = -1;
            float sum6 = 0;
            float sum7 = 0;
            for(int iteration = 0; iteration < 50; iteration++){
                //cout << "*" <<iteration << "*" << endl;
                // variáveis de decisao

                float opt = 0;
                int index = 0;
               #pragma omp parallel for schedule(static) reduction(+:opt)
                for(int p=0; p < S; p=p+sb_size){
                 //cout << p << endl;



                    IloEnv env;
                    IloModel mod(env);
                    IloCplex cplex(mod);
                    IloNumVarArray startVar(env);
                    IloNumArray startVal(env);

                    IloArray<IloNumVarArray> C(env, T);
                    IloArray<IloArray<IloNumVarArray> > X(env, N);
                    IloArray<IloArray<IloNumVarArray> > B(env, N);
                    IloArray<IloArray<IloNumVarArray> > V(env, N);



                   //cout << "1" << endl;
                    for(int j = 0; j < T; j++)
                        C[j] = IloNumVarArray(env, sb_size, 0, 1, ILOINT);

                   // cout << "2" << endl;
                    for(int i= 0; i < N; i++){
                        X[i] = IloArray<IloNumVarArray>(env, T+1);
                        B[i] = IloArray<IloNumVarArray>(env, T+1);
                        V[i] = IloArray<IloNumVarArray>(env, T+1);

                        for(int j= 0; j <= T; j++){
                            X[i][j] = IloNumVarArray(env, sb_size);
                            B[i][j] = IloNumVarArray(env, sb_size);
                            V[i][j] = IloNumVarArray(env, sb_size);

                            for(int k=0; k < sb_size; k++){
                                X[i][j][k] = IloNumVar(env, 0, IloInfinity, ILOFLOAT);
                                B[i][j][k] = IloNumVar(env, 0, IloInfinity, ILOFLOAT);
                                V[i][j][k] = IloNumVar(env, 0, IloInfinity, ILOFLOAT);

                            }

                        }

                    }



                    //===================================================
                    // Maximize P[i][j][k]*X[i][j][k]
                    //====================================================
                    //cout << "3"<< endl;
                    IloExpr fo(env);
                    IloExpr penalty1(env);
                    IloExpr penalty2(env);
                    for(int i=0; i < N; i++){
                        for(int k=0; k < sb_size; k++){
                            fo += float(1.0/S)*P[i][T][p+k]*X[i][T][k];

                        }

                        if(iteration != 0){



                            for(int k = 0; k < sb_size; k++)
                                for(int j = 0; j < T-sb_stages; j++){


                                    if(lambda1[i][j][p] != 0)
                                    penalty1 += lambda1[i][j][p]*(X[i][j][0] - X_at[i][j][p]);

                                    penalty1 += (1/2)*rho*(X[i][j][0]  - 2*X[i][j][0]*X_at[i][j][p] - X_at[i][j][p]*X_at[i][j][p]);


                                }



                        }

                        else{

                            penalty1 += 0;
                           // penalty2 += 0;

                        }


                    }

                    IloAdd(mod, IloMaximize(env, fo - penalty1));
                    penalty1.end();
                    fo.end();


                    //========================================================
                    //  sum X[i][0][k]*P[i][0][k] forall i=1,...,N, k=1,...,S
                    //========================================================
                 //   cout << "4" << endl;

                    for(int k=0; k < sb_size; k++){
                        IloExpr constraint1(env);
                        for(int i=0; i <N; i++){
                            constraint1 += P[i][0][p+k]*X[i][0][k];

                            }
                        mod.add(constraint1 == Q);
                        constraint1.end();

                    }


                    //======================================================================================
                    // X[i][j][k] = X[i][j-1][k] + B[i][j][k] - V[i][j][k] for i=1,...,N,j=1,...,T,,k=1,...S
                    //======================================================================================
                  //  cout << "5" << endl;
                    for(int i=0; i < N; i++){

                        for(int j=0; j <= T; j++){

                            for(int k=0; k < sb_size; k++){
                                IloExpr constraint2(env);

                                if(j==0){

                                    constraint2 += X[i][j][k] - B[i][j][k];
                                    mod.add(V[i][j][k] == 0);
                                }



                                else{

                                    constraint2 += X[i][j][k] - X[i][j-1][k] - B[i][j][k] + V[i][j][k];
                                }

                                mod.add(constraint2 == 0);
                                constraint2.end();

                            }

                        }

                    }

                    //=========================================================================================
                    // sum P[i][j][k]V[i][j][k] - sum P[i][j][k]B[i][j][k] + ft = lt forall j=1,...,T, k=1,...,S
                    //=========================================================================================
                  //  cout << "6" << endl;
                    for(int j = 1; j <= T; j++){


                        for(int k = 0; k < sb_size; k++){

                            IloExpr constraint3_1(env);
                            IloExpr constraint3_2(env);
                            for(int i = 0; i < N; i++){

                                constraint3_1 += P[i][j][p+k]*B[i][j][k];
                                constraint3_2 += P[i][j][p+k]*V[i][j][k];
                            }
                            mod.add(constraint3_2 - constraint3_1 + ft[j-1] == lt[j-1]);
                            constraint3_1.end();
                            constraint3_2.end();

                        }

                    }

                    //============================================================================================
                    // X[i][j][k]*P[i][j][k] < pi * sum X[i][j][k]*P[i][j][k] forall i=1,...,N,j=1,...T, k=1,...,S
                    // ===========================================================================================
                   // cout << "7" << endl;
                    for(int i = 0; i < N; i++){

                        for(int j = 1; j <= T; j++){

                            for(int k = 0; k < sb_size; k++){
                                IloExpr constraint4(env);
                                for(int m = 0; m < N; m++){
                                    constraint4 += X[m][j][k]*P[m][j][p+k];
                                }
                                mod.add(X[i][j][k]*P[i][j][p+k] <= pi[i]*constraint4);
                                constraint4.end();

                            }


                        }

                    }

                    //==================================================================
                    //K(Lt - Ft) - sum P[i][j][k]*X[i][j][k] <= MC[j][k] forall j=1,...,T, k=1,...,S
                    //==================================================================
                  // cout << "8" << endl;

                    for(int j = 1; j <= T; j++){

                        for(int k = 0; k < sb_size; k++){
                            IloExpr constraint5(env);
                            for(int i = 0; i < N; i++){
                                constraint5 += X[i][j][k]*P[i][j][p+k];
                            }

                            mod.add(K*(Lt[j-1] - Ft[j-1]) - constraint5 - M*C[j-1][k] <= 0);
                            constraint5.end();

                        }

                    }

                    //============================================================
                    // sum C[j][k] <= 2 forall j=1,...,T-2, k=1,...,S
                    //=============================================================
                //   cout << "9" << endl;

                    for(int j=0; j < T-2; j++){

                        for(int k=0; k < sb_size; k++){

                            IloExpr constraint6(env);
                            for(int t = 0; t <= 2; t++){
                                constraint6 += C[j+t][k];
                            }

                            mod.add(constraint6 <= 2);
                            constraint6.end();

                        }

                    }


                for(int i=0; i < N; i++){
                    int block_size = sb_size;
                    for(int j=0; j < T; j++){

                            if(j < T - sb_stages){

                                for(int k = 0; k < sb_size-1; k++){

                                    mod.add(X[i][j][k] - X[i][j][k+1] == 0);
                                }
                            }


                            else{

                                int cont = 1;
                                for(int k = 0; k < sb_size-1; k++){

                                     //cout << k << endl;
                                     if(cont == block_size){
                                        cont = 1;
                                        continue;

                                    }

                                    mod.add(X[i][j][k] - X[i][j][k+1] == 0);
                                    cont = cont + 1;
                                }

                        block_size = block_size/branching;
                        }

                    }


                }


                    for(int j = 0; j < T; j++){

                        for(int k = 0; k < sb_size; k++){

                            int sum = 0;

                            for(int i = 0; i < 5; i++)
                                sum = sum + fix_count[j][p+k][i];

                            if(sum >= 5)
                                C[j][k].setBounds(C1[j][p+k],C1[j][p+k]);



                        }

                    }

                    /*
                    if(iteration != 0)
                        for(int i = 0; i < N; i++)
                            for(int j = 0; j < T; j++){
                                startVar.add(X[i][j][0]);
                                startVal.add(X1[i][j][p]);
                            }*/


                    IloTimer crono(env);// Variável para coletar o tempo
                    cplex.setParam(IloCplex::Param::Threads,1);
                   // crono.start();
                    cplex.setWarning(env.getNullStream());
                    cplex.setOut(env.getNullStream()); // Eliminar os logs do solver
                   // cout << "***" << endl;
                    cplex.solve();
                    //#pragma omp atomic
                   // cout << cont1 << endl;
                    opt += cplex.getObjValue();
                   // crono.stop();



                //    cout << "10" << endl;

                    //a solucao da variavel de decisao X e colocada em matriz
                    for(int i=0; i < N; i++){

                        for(int j=0; j < T; j++)
                            for(int k=0; k < 1; k++)
                                X1[i][j][p+k] = cplex.getValue(X[i][j][k]);



                    }

                //    cout << "11" << endl;
                    for(int j=0; j < T; j++)
                            for(int k=0; k < sb_size; k++){

                                if(C1[j][p+k] != cplex.getValue(C[j][k])){
                                    C1[j][p+k] = cplex.getValue(C[j][k]);
                                    fix_count[j][p+k][cont] = 0;
                                }

                                else
                                    if(iteration != 0)
                                        fix_count[j][p+k][cont] = 1;

                            }

                    /*
                    for(int j=0; j <= T; j++){

                        output1 << cplex.getValue(X[0][j][0]) << '\t';
                        output2 << cplex.getValue(X[1][j][0]) << '\t';

                    }*/
                   //  cout << "12" << endl;

                  env.end();



                }


                float sum = 0;
                float sum1 = 0;
                //computa criterio de parada
                for(int i = 0; i < 1; i++){

                        for(int k = 0; k < S; k = k + sb_size){

                            for(int j = 0; j < T - sb_stages; j++){

                                sum = sum + (X1[i][j][k] - X_at[i][j][k])*(X1[i][j][k] - X_at[i][j][k]);
                                X_aux[i][j][k] = X_at[i][j][k];
                            }

                        sum1 = sum1 + (1.0/pow(branching, T-sb_stages))*sum;
                        sum = 0;
                        }

                }
              //  cout << "13" << endl;


                float cont1;



                sum1 = sqrt(sum1);



                if(iteration == 0)
                    cont1 = sum1;


              //  cout << "--" << sum1 << "--" << endl;
               // cout << "!!" << float(sum1/cont1) << "!!" << endl;
                last_ite = iteration;
                last_opt = opt;
                cout << opt << endl;
                output2 << opt << endl;
                //testa o criterio de parada
                if(float(sum1/cont1) <= 0.05){


                    //output2 << opt << endl;
                    break;
                }

                float aux = 0;
               // cout << "14" << endl;
                for(int i=0; i < N; i++){

                        for(int j=0; j < T - sb_stages; j++){

                            for(int k = 0; k < S; k = k + sb_size){

                                for(int m = floor(k/pow(branching, T -j))*pow(branching, T  -j); m < floor(k/pow(branching, T -j))*pow(branching, T - j) + pow(branching, T-j); m = m + sb_size){

                                    aux = aux + (1.0/pow(branching,T-j))*X1[i][j][m];


                                }


                            X_at[i][j][k] = sb_size*aux;
                            aux = 0;


                            }


                        }


                }
/*
                for(int i = 0; i < N; i++){

                        for(int k = 0; k < S;  k= k+ sb_size){

                            for(int j = 0; j < T - sb_stages; j++){

                                output1 << X1[i][j][k] - X_at[i][j][k] << '\t';
                            }
                        output1 << endl;
                        }
                output1 << endl;
                }*/
                //cout << "14" << endl;

                float sum2 = 0;
                float sum3 = 0;
                float sum4 = 0;
                float sum5 = 0;
                for(int i = 0; i < N; i++){

                    for(int k = 0; k < S; k = k + sb_size){
                       // #pragma omp parallel for
                        for(int j = 0; j < T - sb_stages; j++){

                            lambda1[i][j][k] = lambda1[i][j][k] + rho*(X1[i][j][k]- X_at[i][j][k]);
                            if(i==0){
                                sum2 = sum2 + (X1[i][j][k]- X_at[i][j][k])*(X1[i][j][k]- X_at[i][j][k]);
                                sum4 = sum4 + (X_at[i][j][k]- X_aux[i][j][k])*(X_at[i][j][k]- X_aux[i][j][k]);
                                }

                            }
                        if(i==0){
                            sum3 = sum3 + (1.0/pow(branching, T-sb_stages))*sum2;
                            sum5 = sum5 + (1.0/pow(branching, T-sb_stages))*sum4;
                            sum2 = 0;
                            sum4 = 0;
                            //sum3 = 0;
                            }
                        }

                    }

            /*
            if(sum6 - sum3 > 0){
                rho = rho/2;

            }

            if(sum7 - sum5 > 0){
                rho = 2*rho;

            }

            sum6 = sum3;
            sum7 = sum5;

        /*
            cout << "!" << cont << endl;
            for(int j = 0; j < T; j++){
                for(int k = 0; k < S; k++){
                    for(int t = 0; t < 5; t++)
                        cout << fix_count[j][k][t] << '\t';
                cout << endl;
                }
            cout << endl;
            }*/



            if (cont < 4)
                cont = cont + 1;

            else
                cont = 0;

            }



        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);
        cout << duration.count() << endl;
        output1 << "\t" << last_opt << "\t" << last_ite + 1 << "\t" << duration.count() << endl;
        output2 << duration.count() << endl;


        }



    arq.close();
    output1.close();
    output2.close();


    }

     catch(IloException& exc){

        cerr << "Error:" << exc << endl;

    }

    return 0;

}


