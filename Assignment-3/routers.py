from schemas import AgentState

class Routers:
    
    def __init__(self):
        pass

    def validation_router(self, state: AgentState):
            print("=========================================== [ Retry Validation Routing in Progress ] ===========================================")
            
            validation_status = state["validation_status"]
            last_node = state["last_node"]
            
            if validation_status == "passed" or validation_status == "max_retries":
                print("Routed To -> Final Output")
                print("=========================================== [ Retry Validation Routing Completed ] ===========================================\n")
                return "Output"
            elif validation_status == "retry":
                if last_node == "RAG":
                    print("Routed To -> Retry RAG Call")
                    print("=========================================== [ Retry Validation Routing Completed ] ===========================================\n")
                    return "RETRY_RAG"
                elif last_node == "LLM":
                    print("Routed To -> Retry LLM Call")
                    print("=========================================== [ Retry Validation Routing Completed ] ===========================================\n")
                    return "RETRY_LLM"
                
    def simple_router(self, state : AgentState):

        print("=========================================== [ Routing in Progress ] ===========================================")

        topic = state['messages'][-1]

        if topic == "Data Structures":
            print("Routed To --> RAG CALL")
            print("=========================================== [ Successfully Routed ] ===========================================\n")
            return "RAG Call"
        elif topic == "Web Search":
            print("Routed To --> Web CALL")
            print("=========================================== [ Successfully Routed ] ===========================================\n")
            return "Web Call"
        else:
            print("Routed To --> LLM CALL")
            print("=========================================== [ Successfully Routed ] ===========================================\n")
            return "LLM Call"