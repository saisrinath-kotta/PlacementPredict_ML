class Table extends Thread{
    int n;
    Table(int n){
    this.n = n;
    }

    public void run(){
        System.out.println("\n Table of " + n);

        for(int i = 1;i<=10;i++){
            System.out.println(n+"x" + i+"="+(n*i));
        }
    }
}

class TwoTables{
    public static void main(String[]args) throws Exception{
        Table t1 = new Table(5);
        Table t2 = new Table(10);

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("\n Both tables completed");
    }
}
