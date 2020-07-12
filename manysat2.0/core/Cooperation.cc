/*****************************************************************************************[Cooperation.h]
Copyright (c) 2008-20011, Youssef Hamadi, Saïd Jabbour and Lakhdar Saïs
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************************/
#include "core/Cooperation.h"


namespace Minisat {
  
  /*_________________________________________________________________________________________________
    |
    |  exportExtraUnit : ()  ->  [void]
    |  Description : manage export Extra Unit Clauses 
    |________________________________________________________________________________________________@*/
  
  void Cooperation::exportExtraUnit(Solver* s, Lit unit){
    
    int id = s->threadId;
    
    for(int t = 0; t < nbThreads; t++){
      
      if(t == id) continue;
      int ind = tailExtraUnits[id][t];
      if(((ind+1)%MAX_EXTRA_UNITS) == headExtraUnits[id][t]) continue;
      
      extraUnits[id][t][ind++].x = unit.x;
      
      if(ind == MAX_EXTRA_UNITS) ind = 0;
      tailExtraUnits[id][t] = ind;
    }
  }

  /*_________________________________________________________________________________________________
    |
    |  importExtraUnits : ()  ->  [void]
    |  Description : store unit extra clause in the vector unit_learnts
    |________________________________________________________________________________________________@*/
  
  void Cooperation::importExtraUnits(Solver* s, vec<Lit>& unit_learnts){ 
    
    int id = s->threadId;
    
    for(int t = 0; t < nbThreads; t++){
      
      if(t == id) 
	continue;
      
      int head = headExtraUnits[t][id];
      int tail = tailExtraUnits[t][id];
      if(head == tail) 
	continue;
      
      int end = tail; 
      if(tail < head) end = MAX_EXTRA_UNITS;
      
      for(int i = head; i < end; i++)
	storeExtraUnits(s, t, extraUnits[t][id][i], unit_learnts);
      
      if(tail < head)
	for(int i = 0; i < tail; i++)
	  storeExtraUnits(s, t, extraUnits[t][id][i], unit_learnts);
      
      head = tail;
      if(head == MAX_EXTRA_UNITS) head = 0;
      headExtraUnits[t][id] = head;
    }
  }
    
  /*_________________________________________________________________________________________________
    |
    |  importExtraUnits : ()  ->  [void]
    |  Description : manage import Extra Unit Clauses 
    |________________________________________________________________________________________________@*/
  
  void Cooperation::importExtraUnits(Solver* s){ 
    
    int id = s->threadId;
    
    for(int t = 0; t < nbThreads; t++){
      
      if(t == id) 
	continue;
      
      int head = headExtraUnits[t][id];
      int tail = tailExtraUnits[t][id];
      if(head == tail) 
	continue;
      
      int end = tail; 
      if(tail < head) end = MAX_EXTRA_UNITS;
      
      for(int i = head; i < end; i++)
	uncheckedEnqueue(s, t, extraUnits[t][id][i]);
      
      if(tail < head)
	for(int i = 0; i < tail; i++)
	  uncheckedEnqueue(s, t, extraUnits[t][id][i]);
      
      head = tail;
      if(head == MAX_EXTRA_UNITS) head = 0;
      headExtraUnits[t][id] = head;
    }
  }
  
  
  /*_________________________________________________________________________________________________
    |
    |  uncheckedEnqueue : ()  ->  [void]
    |  Description: Enqueue the unit literals at level 0
    |________________________________________________________________________________________________@*/
  
  void Cooperation::uncheckedEnqueue(Solver* s, int t, Lit l){
    
    if(s->value(l) == l_False) answers[s->threadId] = l_False;
    
    if(s->value(l) != l_Undef) return;
    s->uncheckedEnqueue(l);
    nbImportedExtraUnits[s->threadId]++;
    pairwiseImportedExtraClauses[t][s->threadId]++;
  }

  /*_________________________________________________________________________________________________
    |
    |  uncheckedEnqueue : ()  ->  [void]
    |  Description: Enqueue the unit literals at level 0
    |________________________________________________________________________________________________@*/
  
  void Cooperation::storeExtraUnits(Solver* s, int t, Lit l, vec<Lit>& unitLits){
    unitLits.push(l);
    nbImportedExtraUnits[s->threadId]++;
    pairwiseImportedExtraClauses[t][s->threadId]++;
  }
  
  /*_________________________________________________________________________________________________
    |
    |  exportExtraClause : ()  ->  [void]
    |  Description: each time a new clause is learnt, it is exported to other threads
    |________________________________________________________________________________________________@*/
  
  
  void Cooperation::exportExtraClause(Solver* s, vec<Lit>& learnt){
    
    int id = s->threadId;
    
    for(int t = 0; t < nbThreads; t++){
      
      if((t == id) || (learnt.size() > pairwiseLimitExportClauses[id][t])) 
	continue;
      
      int ind = tailExtraClauses[id][t];
      if(((ind+1) % MAX_EXTRA_CLAUSES) == headExtraClauses[id][t]) 
	continue;
      
      extraClauses[id][t][ind]    = new Lit [learnt.size() + 1];
      extraClauses[id][t][ind][0] = mkLit(learnt.size());
      
      for(int j = 0; j < learnt.size(); j++)
	extraClauses[id][t][ind][j + 1] = learnt[j];
      
      ind++;
      if(ind == MAX_EXTRA_CLAUSES) ind = 0;
      tailExtraClauses[id][t] = ind;
    }
  }		      
  
  
  /*_________________________________________________________________________________________________
    |
    |  exportExtraClause : ()  ->  [void]
    |  Description: each time a new clause is learnt, it is exported to other threads
    |________________________________________________________________________________________________@*/
  
  void Cooperation::exportExtraClause(Solver* s, Clause& c){
    
    int id = s->threadId;
    
    for(int t = 0; t < nbThreads; t++){
      
      if((t == id) || (c.size() > pairwiseLimitExportClauses[id][t])) 
	continue;
      
      int ind = tailExtraClauses[id][t];
      if(((ind+1) % MAX_EXTRA_CLAUSES) == headExtraClauses[id][t]) 
	continue;
      
      extraClauses[id][t][ind]    = new Lit [c.size() + 1];
      extraClauses[id][t][ind][0] = mkLit(c.size());
      
      for(int j = 0; j < c.size(); j++)
	extraClauses[id][t][ind][j + 1] = c[j];
      
      ind++;
      if(ind == MAX_EXTRA_CLAUSES) ind = 0;
      tailExtraClauses[id][t] = ind;
    }
  }		      
  
  /*_________________________________________________________________________________________________
    |
    |  importExtraClauses : ()  ->  [void]
    |  Description: import Extra clauses sent from other threads
    |_______________________________________________________________________________________________@*/
  
  void Cooperation::importExtraClauses(Solver* s){
    
    int id = s->threadId;
    
    for(int t = 0; t < nbThreads; t++){
      
      if(t == id) 
	continue;
      
      int head = headExtraClauses[t][id];
      int tail = tailExtraClauses[t][id];
      if(head == tail) 
	continue;
      
      int end = tail; 
      if(tail < head) end = MAX_EXTRA_CLAUSES;
      
      for(int i = head; i < end; i++)
	addExtraClause(s, t, extraClauses[t][id][i]);
      
      if(tail < head)
	for(int i = 0; i < tail; i++)
	  addExtraClause(s, t, extraClauses[t][id][i]);
      
      head = tail;
      if(head == MAX_EXTRA_CLAUSES) head = 0;
      headExtraClauses[t][id] = head;
    }
  }    

	
  /*_________________________________________________________________________________________________
    |
    |  addExtraClause : ()  ->  [void]
    |  Description: build a clause from the learnt Extra Lit*
    |  watch it correctly, test basic cases distinguich other cases during watching process 
    |________________________________________________________________________________________________@*/
  
  
  void Cooperation::addExtraClause(Solver* s, int t, Lit* lt){
    
    vec<Lit>  extra_clause;           
    int       extra_backtrack_level = 0;
    int id   = s->threadId;
    int size = var(lt[0]);
    int    wtch = 0;
    Lit q = lit_Undef;
    
    for(int i = 1, j = 0; i < size+1; i++) {
      Lit q = lt[i];
      if(s->value(q) == l_False && s->level(var(q)) == 0)
	continue;
      extra_clause.push(q);
      if(s->value(q) != l_False && wtch < 2){
	extra_clause[j] = extra_clause[wtch];
	extra_clause[wtch++] = q;
      }else if(s->level(var(q)) >= extra_backtrack_level)
	extra_backtrack_level = s->level(var(q));
      j++;
    }
    
    
    //conflict clause at level 0 --> formula is UNSAT
    if(extra_clause.size() == 0) 
      setAnswer(id, l_False);
    
    // case of unit extra clause
    else if(extra_clause.size() == 1){
      s->cancelUntil(0);
      if(s->value(extra_clause[0]) == l_Undef){
	s->uncheckedEnqueue(extra_clause[0]);
	CRef cs = s->propagate();
	if(cs != CRef_Undef) answers[id] = l_False;
      }
    }
    
    else {
      // build clause from lits and add it to learnts(s) base
      CRef cr = s->addExtraClause(extra_clause);
      
      // Case of Unit propagation: literal to propagate or bad level  
      if(wtch == 1 && (s->value(extra_clause[0]) == l_Undef || extra_backtrack_level < s->level(var(extra_clause[0])))){
	s->cancelUntil(extra_backtrack_level);
	s->uncheckedEnqueue(extra_clause[0], cr);
      }
      
      // Case of Conflicting Extra Clause --> analyze that conflict
      else if(wtch == 0) {
	extra_clause.clear();
	s->cancelUntil(extra_backtrack_level);
	
	s->analyze(cr, extra_clause, extra_backtrack_level);	
	s->cancelUntil(extra_backtrack_level);
	
	// analyze lead to unit clause
	if(extra_clause.size() == 1){				
	  assert(extra_backtrack_level == 0);
	  if(s->value(extra_clause[0]) == l_Undef)s->uncheckedEnqueue(extra_clause[0]);
	}
	// analyze lead to clause with size > 1
	else {
	  CRef cs = s->addExtraClause(extra_clause);
	  s->uncheckedEnqueue(extra_clause[0], cs);
	  //exportExtraClause(s, extra_clause);		
	}
      }
    }
    
    nbImportedExtraClauses[id]++;
    pairwiseImportedExtraClauses[t][id] ++;
    
    delete [] lt;
  }		      
  



  /*_________________________________________________________________________________________________
    |
    |  addExtraClause : ()  ->  [void]
    |  Description: watch lits correctly and testing basic cases and
    |  build a clause from the learnt Extra Lit* 
    |________________________________________________________________________________________________@*/
  
  void Cooperation::addExtraClause1(Solver* s, int t, Lit* lt){
    
    int id = s->threadId;
    int size = var(lt[0]);
    
    vec<Lit> lits;
    int j = 0;
    int k = 0;
    Lit q        = lit_Undef;
    int maxLevel = 0;
    
    for(int i=1; i < size+1; i++) {
      Lit q = lt[i];
      if(s->value(q) == l_False && s->level(var(q)) == 0)
	continue;
      lits.push(lt[i]);
      if(s->value(q) != l_False){
	lits[k]   = lits[j];
	lits[j++] = q;
      }else if(s->level(var(q)) >= maxLevel)
	maxLevel = s->level(var(q));
      k++;
    }
    
    //conflict clause at level 0
    if(lits.size() == 0) 
      setAnswer(id, l_False);  
    
    // case of unit extra clause 
    else if(lits.size() == 1){
      s->cancelUntil(0);
      if(s->value(lits[0]) == l_Undef)s->uncheckedEnqueue(lits[0]);
    }
    else {
      // build clause from lits
      s->addExtraClause(lits);
    }
    
    nbImportedExtraClauses[id]++;
    pairwiseImportedExtraClauses[t][id]++;
    
    delete [] lt;
  }
    

  /*_________________________________________________________________________________________________
    |
    |  addExtraClause : ()  ->  [void]
    |  Description: build a clause from the learnt Extra Lit*
    |________________________________________________________________________________________________@*/
  
  void Cooperation::addExtraClause2(Solver* s, int t, Lit* lt){
    
    int id = s->threadId;
    int size = var(lt[0]);
    vec<Lit> lits;
    
    for(int i=1; i < size+1; i++) 
      lits.push(lt[i]);
    
    // build clause from lits
    s->addExtraClause(lits);
    
    nbImportedExtraClauses[id]++;
    pairwiseImportedExtraClauses[t][id]++;
    
    delete [] lt;
  }
  
  /*_________________________________________________________________________________________________
    |
    |  updateLimitExportClauses : ()  ->  [void]
    |  update limit size of exported clauses in order to maintain exchange during search:
    |  (MAX_IMPORT_CLAUSES / LIMIT_CONFLICTS) is the percent of clauses that must be shared
    |________________________________________________________________________________________________@*/
  
  
  void Cooperation::updateLimitExportClauses(Solver* s){
    
    int id = s->threadId;
    
    if((int)s->conflicts % LIMIT_CONFLICTS_EVAL != 0) return;
    
    double sumExtImpCls = 0;      
    
    for(int t = 0; t < nbThreads; t++)
      sumExtImpCls += pairwiseImportedExtraClauses[t][id];
    
    switch(ctrl){
      
    case 1 :{	      
      if(sumExtImpCls <= MAX_IMPORT_CLAUSES)
	for(int t = 0; t < nbThreads; t++)
	  pairwiseLimitExportClauses[t][id]+=1;
      else
	for(int t = 0; t < nbThreads; t++)
	  pairwiseLimitExportClauses[t][id]-=1;
      
      for(int t = 0; t < nbThreads; t++)
	pairwiseImportedExtraClauses[t][id] = 0;
      break;
    }
      
      
    case 2 :{
      if(sumExtImpCls <= MAX_IMPORT_CLAUSES)
	for(int t = 0; t < nbThreads; t++)
	  pairwiseLimitExportClauses[t][id] += aimdy / pairwiseLimitExportClauses[t][id];	
      else
	for(int t = 0; t < nbThreads; t++)
	  pairwiseLimitExportClauses[t][id] -= aimdx * pairwiseLimitExportClauses[t][id];
      
      for(int t = 0; t < nbThreads; t++)
	pairwiseImportedExtraClauses[t][id] = 0;
      break;
    }
    }
  }
  
  /*_________________________________________________________________________________________________
    |
    |  printStats : ()  ->  [void]
    |  Description: print statistics of each thread
    |________________________________________________________________________________________________@*/
  
  void Cooperation::printStats(int& id){
    
    uint64_t nbSharedExtraClauses = 0;
    uint64_t nbSharedExtraUnits   = 0;
    uint64_t totalConflicts       = 0;	
    
    for(int t = 0; t < nbThreads; t++){
      
      nbSharedExtraClauses += nbImportedExtraClauses[t];
      nbSharedExtraUnits   += nbImportedExtraUnits  [t];
      totalConflicts += solvers[t].conflicts;
    }
    
    
    printf(" -----------------------------------------------------------------------------------------------------------------------\n");
    printf("| ManySAT 2.2 - all threads statistics           DETERMINISTIC MODE ? %s              initial limit export clauses : %3d |\n", 
	   deterministic_mode == true ? "Y" : "N",
	   limitExportClauses);
    printf("|-----------------------------------------------------------------------------------------------------------------------|\n");
    printf("|  Thread  | winner  |   #restarts   |   decisions   |  #conflicts   |   %% shared cls  |  #extra units | #extra clauses |         \n");
    printf("|----------|---------|---------------|---------------|---------------|-----------------|---------------|----------------|\n");
    
    for(int t = 0; t < nbThreads; t++)
      printf("| %8d | %7s | %13d | %13d | %13d | %13d %% | %13d | %14d |\n", 
	     t, 
	     t==id?"x":" ",
	     (int)solvers[t].starts,
	     (int)solvers[t].decisions, 
	     (int)solvers[t].conflicts, 
	     (int)solvers[t].conflicts==0 ? 0 : (int) (nbImportedExtraUnits[t] + nbImportedExtraClauses[t]) * 100 / (int)solvers[t].conflicts,
	     (int)nbImportedExtraUnits[t], 
	     (int)nbImportedExtraClauses[t]);
    
    
    printf("|----------------------------------------------------|---------------|-----------------|---------------|----------------|\n");
    printf("|                                                    | %13d | %13d %% | %13d | %14d |\n", 
	   (int)totalConflicts, 
	   (int)totalConflicts==0 ? 0 : (int) (nbSharedExtraUnits + nbSharedExtraClauses) * 100 / (int)totalConflicts,	
	   (int)nbSharedExtraUnits,	
	   (int)nbSharedExtraClauses);
    printf(" -----------------------------------------------------------------------------------------------------------------------\n");
    
    Parallel_Info();

  }
  
  
  /*_________________________________________________________________________________________________
    |
    |  printExMatrix : ()  ->  [void]
    |  Description: print the final values of size clauses shared between parwise threads
    |________________________________________________________________________________________________@*/
  
  void Cooperation::printExMatrix(){
    
    printf("\n Final Matrix extra shared clauses limit size \n");
    printf(" ----------------------------------------------------\n");
    for(int i = 0; i < nbThreads; i++){
      printf("|");  
      for(int j = 0; j < nbThreads; j++)
	if(i != j)
	  printf(" e(%d,%d)=%3d  ", i, j, (int)pairwiseLimitExportClauses[i][j]);
	else
	  printf(" e(%d,%d)=%3d  ", i, j, 0);
      printf(" |\n");
    }
    printf(" ----------------------------------------------------\n\n");
  }
  
  
  /*_________________________________________________________________________________________________
    |
    |  Parallel_Info : ()  ->  [void]
    |  Description: more informations about deterministic or non deterministic mode and final matrix print
    |________________________________________________________________________________________________@*/
  
  void Cooperation::Parallel_Info(){
    printf(" DETERMINISTIC_MODE? : %s \n", (deterministic_mode == true ? "YES" : "NO"));
    printf(" CONTROL POLICY?	 : %s  \n", (ctrl != 0 ? (ctrl==1 ? "DYNAMIC (INCREMENTAL  en= +- 1)" : "DYNAMIC (AIMD: en=  en - a x en, en + b/en)") : "STATIC (fixed Limit Export Clauses)"));
    if(ctrl == 0) 
      printf(" LIMIT EXPORT CLAUSES (PAIRWISE) : [identity] x %d\n\n", limitExportClauses);
    else
      printExMatrix();
  }
}
