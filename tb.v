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
    
    initial begin
        Clk = 0;
        pcCLK = 0;
        
        #3000;

        #10 $finish;
    end
endmodule
