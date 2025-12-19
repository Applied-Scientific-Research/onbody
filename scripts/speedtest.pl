#!/usr/bin/perl
#

# which binary to run?
#my $exe = "./onvortgrad3d";
#my $exe = "./onvort2d";
my $exe = "./onvort2d";

# how many particles?
my @nump = (100000);

# desired mean velocity error
my @errors = (1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7);

# bucket sizes
#my @bucket = (32, 64, 128);
my @bucket = (32, 64);
#my @bucket = (32);
#my @bucket = (128);

# define the order range
my @order = (1,2,3,4,5,6,7,8,9);
#my @order = (1,2,3,4);

# how many runs over which to average the error?
my $runs = 5;

# after 6 we really should have something close to the solution
my $maxiters = 10;

my $sep = " ";

# put all results here
open(OUT,">>new.dat") or die "Can't open $!";

# for a given set of the above variables, find theta to minimize CPU time
foreach $thisnump (@nump) {
 foreach $desirederror (@errors) {
  my $logdesired = log($desirederror);
  foreach $thisbucket (@bucket) {
   foreach $thisorder (@order) {

    # find the rifa that returns the right error!
    my $rifa = -1.0;
    my $noupperbound = 1;
    my $nolowerbound = 1;
    my $errordiff = 1.0;
    my $numiters = 0;

    print "Finding theta for $thisnump $desirederror $thisbucket $thisorder\n";

    # run until we're within 3% of desired error
    while ($errordiff > 0.03*$desirederror and $numiters < $maxiters) {

      # pick a rifa to use this iteration
      if ($rifa < 0.0) {
        $rifa = 1.0;
        $urifa = 1.0;
        $lrifa = 1.0;
      } elsif ($thiserror > $desirederror) {
        # increase rifa
        #print "  increase rifa\n";
        $nolowerbound = 0;
        $lrifa = $rifa;
        $lerror = $logthiserror;
        if ($noupperbound == 1) {
          # if we don't have an upper bound, increase rifa
          $rifa = $rifa * 1.5;
          $urifa = $rifa;
        } else {
          # if we do, pick the halfway point
          #$rifa = 0.5*($rifa + $urifa);
          # or, use log of error to pick better point
          $rifa = $lrifa + ($urifa - $lrifa)*($logdesired-$lerror)/($uerror-$lerror);
        }
        #print "  rifa is now $rifa\n";
        #exit(0);
      } else {
        # decrease rifa
        #print "  decrease rifa\n";
        $noupperbound = 0;
        $urifa = $rifa;
        $uerror = $logthiserror;
        if ($nolowerbound == 1) {
          # if we don't have an lower bound, decrease rifa
          $rifa = $rifa / 1.5;
          $lrifa = $rifa;
        } else {
          # if we do, pick the halfway point
          #$rifa = 0.5*($lrifa + $rifa);
          # or, use log of error to pick better point
          $rifa = $lrifa + ($urifa - $lrifa)*($logdesired-$lerror)/($uerror-$lerror);
        }
        #print "  rifa is now $rifa\n";
        #exit(0);
      }
      #print "  is upper? $noupperbound   is lower? $nolowerbound\n";
      print "  lower $lrifa   trying $rifa   upper $urifa\n";

      # run the command a few times
      unlink "tempout";
      $command = "${exe} -n=$thisnump -b=$thisbucket -t=$rifa -o=$thisorder >> tempout";
      #print $command."\n";
      for (my $irun=0; $irun<$runs; $irun=$irun+1) { system $command; }

      # parse tempout

      # look for velocity errors
      #@lines = `grep \"error in treecode3\" tempout`;
      @lines = `grep \"error in fastsumm\" tempout`;
      # find the mean of the five errors
      $maxsum = 0.0;
      $meansum = 0.0;
      #print @lines;
      foreach (@lines) {
        @junk = split;
        $maxsum += @junk[4];
        $meansum += @junk[6];
      }
      $maxsum /= $runs;
      $meansum /= $runs;
      # rename it to something useful
      $thismax = $maxsum;
      $thiserror = $meansum;
      # avoid problems taking log of zero
      if ($thiserror > 1.e-8) {
        $logthiserror = log($thiserror);
      } else {
        $logthiserror = log(1.e-8);
      }
      #print $maxsum." ".$thiserror."\n";
      print "    returned mean vel error ".$thiserror."\n";

      # how close is this error to the desired error?
      $errordiff = $desirederror - $thiserror;
      if ($errordiff < 0.0) { $errordiff = -$errordiff; }
      #print "  errordiff ".$errordiff."\n";

      # make sure we don't iterate too much
      $numiters++;
    }

    # we like the error, so let's keep the settings
    # look for total time now
    #@lines = `grep \"treecode3 total\" tempout`;
    @lines = `grep \"fast total\" tempout`;
    $treetime = 0.0;
    foreach (@lines) {
      $_ =~ s/\[//g;
      $_ =~ s/\]//g;
      @junk = split;
      $treetime += @junk[2];
    }
    $treetime /= $runs;

    @lines = `grep \"onbody naive\" tempout`;
    $dirtime = 0.0;
    foreach (@lines) {
      $_ =~ s/\[//g;
      $_ =~ s/\]//g;
      @junk = split;
      $dirtime += @junk[2];
    }
    $dirtime /= $runs;
    print $treetime." ".$dirtime."\n";


    # print the result
    $res = $thisnump.$sep.$thisbucket.$sep.$thisorder.$sep.$rifa.$sep.$maxsum.$sep.$thiserror.$sep.$treetime.$sep.$dirtime."\n";
    #print $res;
    print "  errors, times: $thismax / $thiserror  $treetime / $dirtime\n";
    print OUT $res;

   } # end foreach @order
  } # end foreach @bucket
 } # end foreach @errors
} # end foreach @nump

