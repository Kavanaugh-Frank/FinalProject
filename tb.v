`timescale 1ns/1ps

module cpu_testbench();
    // Clock and reset signals
    reg pcCLK, Clk;
    
    // Instantiate the CPU
    cpu DUT (
        .pcCLK(pcCLK),
        .Clk(Clk)
    );
    
    // Clock generation
    always begin
        #5 Clk = ~Clk;
        #7 pcCLK = ~pcCLK;
    end
    
    // Initial block to display register values and simulate
    initial begin
        // Initialize clocks
        Clk = 0;
        pcCLK = 0;z

        // Infinite loop to display the values of $s0 and $s1 (R16 and R17)
        forever begin
            #10; // Wait 10ns between updates
        end
    end
endmodule
