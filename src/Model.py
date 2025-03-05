import Globals, DataHandling
from Firm import Firms
from Worker import Workers
from Space import Space
import Neural_Network
import argparse




import random
from datetime import datetime

class ModelBuilder:
    def __init__(self):

        self.firm_productivity = 30
        self.epsilon_init = 0.8
        self.num_strat = 26
        self.num_positions = 104  # Number of positions on Salop circle


        if Globals.num_firms == 2:
            self.lower_limit_strat = 2.0
            self.grain_strat = 0.8
        elif Globals.num_firms == 3:
            self.lower_limit_strat = 1.32
            self.grain_strat = 1.09 
        elif Globals.num_firms == 4:
            self.lower_limit_strat = 1.37
            self.grain_strat = 1.23 
        elif Globals.num_firms == 5:
            self.lower_limit_strat =0.78
            self.grain_strat = 1.32     

        

    def build(self, space):
        
        # Set wage list
        self.set_wage_list()

        # Initialize firms and workers
        self.initialize_firms(space)
        self.initialize_workers(space)

        # Arrange agents in the space and grid
        self.arrange_agents(space)

        return space

    def set_wage_list(self):
        self.wage_list = [
            round(self.lower_limit_strat + i * self.grain_strat, 2)
            for i in range(self.num_strat)
            if round(self.lower_limit_strat + i * self.grain_strat, 2) <= self.firm_productivity
        ]

    def initialize_firms(self, context):
        position_firm = 0
        for i in range(Globals.num_firms):
            productivity = self.firm_productivity
            if Globals.asymetric_productivities:
                productivity *= 1 - Globals.delta_productivity if i % 2 == 0 else 1 + Globals.delta_productivity

            context.add_object(Firms(
                space = context,
                firm_id=i,
                position=position_firm,
                wage_offer=self.wage_list[-1],
                productivity=productivity,
                wage_list = self.wage_list
            ))

            position_firm += self.num_positions // Globals.num_firms

    def initialize_workers(self, context):
        for i in range(Globals.num_workers):
            context.add_object(Workers(
                space = context,
                worker_id=i,
                position=i,
                num_positions=self.num_positions,
            ))

    def arrange_agents(self, context):
        for obj in context:
            if isinstance(obj, Workers):
                obj.position = obj.worker_id
            if isinstance(obj, Firms):
                obj.position = obj.firm_id * (self.num_positions // Globals.num_firms)



    def set_simulation_scenario(self, simulation_scenario):
        ''' Scenario 1 and 2: model_type = 0 -> Take it or leave it model
                Scenario 1: With Experience Replay
                Scenario 2: Without Experience Replay

            Scenario 3 and 4: model_type = 1 -> Bidding model
                Scenario 3: With Experience Replay
                Scenario 4: Without Experience Replay
        
        '''

        if simulation_scenario == 1: 

            Globals.model_type == 0 
            Globals.MINI_BATCH_SIZE = 32
            Globals.MEMORY_SIZE = 100000
            Globals.FREQ_UPDATE_TARGETNET =10000

        elif simulation_scenario == 2: 

            Globals.model_type == 0 
            Globals.MINI_BATCH_SIZE = 1
            Globals.MEMORY_SIZE = 1
            Globals.FREQ_UPDATE_TARGETNET =1

        elif simulation_scenario == 3: 

            Globals.model_type == 1 
            Globals.MINI_BATCH_SIZE = 32
            Globals.MEMORY_SIZE = 100000
            Globals.FREQ_UPDATE_TARGETNET =10000
        
        elif simulation_scenario == 4: 

            Globals.model_type == 1 
            Globals.MINI_BATCH_SIZE = 1
            Globals.MEMORY_SIZE = 1
            Globals.FREQ_UPDATE_TARGETNET =1



# Main execution
if __name__ == "__main__": 

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Set the scenario for the AI tool.")
    parser.add_argument(
        "--set_simulation_scenario",
        type=int,
        default=Globals.set_simulation_scenario,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )



    # Set up command-line arguments
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Globals.LEARNING_RATE,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )


     # Set up command-line arguments
    
    parser.add_argument(
        "--beta",
        type=float,
        
        default=Globals.beta,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )



    
    parser.add_argument(
        "--effort",
        type=float,
        default=Globals.effort,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )



    
    parser.add_argument(
        "--random_productivity",
        type=int,
        choices=[1, 0],
        default= Globals.random_productivity,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )

    
    parser.add_argument(
        "--num_firms",
        type=int,
        default=Globals.num_firms,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )


    
    parser.add_argument(
        "--delta_productivity",
        type=float,
        default=Globals.delta_productivity,
        help="Choose the scenario: 1 -> random letter selection, 2 -> AI tool-based selection (default: 2)"
    )


    valid_ranges = [(50000, 55000), (100000, 105000), (150000, 155000)]


    args = parser.parse_args()
    Globals.set_simulation_scenario = args.set_simulation_scenario
    Globals.LEARNING_RATE = args.learning_rate
    Globals.beta = args.beta
    Globals.effort = args.effort
    Globals.num_firms = args.num_firms
    Globals.delta_productivity = args.delta_productivity

    if args.random_productivity == 1:
        Globals.random_productivity = True
    else:
        Globals.random_productivity = False


    if Globals.beta>0.01:
        Globals.beta = Globals.beta*1e-5
    

    modelBuilder = ModelBuilder()
    modelBuilder.set_simulation_scenario(Globals.set_simulation_scenario)
    space = Space()
    space = modelBuilder.build(space)



    firm_list = []
    worker_list = []


    

    for obj in space:
        if isinstance(obj, Firms):
            firm_list.append(obj)
        elif isinstance(obj, Workers):
            worker_list.append(obj)
  

    for iteration in range(Globals.max_iterations):
        #print(f"Iteration   {iteration}")

        for obj in space:
            obj.update_iteration(iteration)

        if Globals.model_type == 0:   # Take-it or leave-it setup

            for firm in firm_list:
                firm.firing()
                firm.set_productivity()
                firm.set_current_state()
                
            for firm in firm_list:
                firm.wage_offer_method()


            
            for worker in worker_list:
                
                worker.applying_takeit()

            for firm in firm_list:
                firm.hiring()
                firm.calculate_profits()
                firm.set_next_state()
                if iteration >= Globals.learning_start:
                    firm.training()

                    if iteration % Globals.FREQ_UPDATE_TARGETNET == 0:
                
                        Neural_Network.copy_weights(firm.policy_net, firm.target_net)
                if Globals.track_data_for_table_1 and any(start < iteration <= end for start, end in valid_ranges):
                    firm.compute_av_greedy_wage_for_analysis()
                #print(f"Greedy wage Firm {firm.firm_id}:    {firm.greedy_wage}")

        elif Globals.model_type == 1:   # Bidding setup

            for firm in firm_list:
                firm.firing()
                firm.set_productivity()
            
            for worker in worker_list:
                worker = obj
                worker.applying_bid()

            for firm in firm_list:
                firm.set_current_state()
            
            for firm in firm_list:
                firm.wage_offer_method()
            
            for worker in worker_list:
                worker.workers_accepts()
                    
            for firm in firm_list:
                firm.hiring()
                firm.calculate_profits()
                firm.set_next_state()
                if iteration >= Globals.learning_start:
                    firm.training()

                    if iteration % Globals.FREQ_UPDATE_TARGETNET == 0:
                
                        Neural_Network.copy_weights(firm.policy_net, firm.target_net)
                if Globals.track_data_for_table_1 and any(start < iteration <= end for start, end in valid_ranges):
                    firm.compute_av_greedy_wage_for_analysis()
                #print(f"Greedy wage Firm {firm.firm_id}:    {firm.greedy_wage}")


        if iteration == 0:

            DataHandling.write_firm_to_csv(iteration, firm_list, file_name="firms_output.csv", append=False)
            DataHandling.write_worker_aggregates_to_csv(iteration, worker_list, file_name="worker_aggregates.csv", append=False)

        
        
        elif iteration > Globals.data_store_freq and iteration < Globals.max_iterations - 1000 and iteration % Globals.data_store_freq == 0:
    
            DataHandling.write_firm_to_csv(iteration, firm_list, file_name="firms_output.csv", append=True)
            DataHandling.write_worker_aggregates_to_csv(iteration, worker_list, file_name="worker_aggregates.csv", append=True)

        elif  Globals.track_data_for_table_1 and any(start < iteration <= end for start, end in valid_ranges):

            DataHandling.write_firm_to_csv(iteration, firm_list, file_name="firms_output.csv", append=True)
            DataHandling.write_worker_aggregates_to_csv(iteration, worker_list, file_name="worker_aggregates.csv", append=True)

        elif iteration >= Globals.max_iterations - 1000: 
            
            DataHandling.write_firm_to_csv(iteration, firm_list, file_name="firms_output.csv", append=True)
            DataHandling.write_worker_aggregates_to_csv(iteration, worker_list, file_name="worker_aggregates.csv", append=True)


        if Globals.save_special_data_for_single_run_analysis and iteration >= 50000 and iteration <=60000 and iteration %10 == 0:
            if iteration==50000:
                print("Start collecting data")

                DataHandling.save_q_values_over_time(iteration, firm_list, file_name="q_values_over_time.csv", append=False)
                DataHandling.save_firm_performance(iteration, firm_list, file_name="firm_performance.csv", append=False)
            else:
                DataHandling.save_q_values_over_time(iteration, firm_list, file_name="q_values_over_time.csv", append=True)
                DataHandling.save_firm_performance(iteration, firm_list, file_name="firm_performance.csv", append=True)

            if iteration ==60000:
                 print("Finish collecting data")

            



    