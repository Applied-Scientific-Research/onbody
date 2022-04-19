#!/usr/bin/perl
#

$sep = " ";

# put all results here
open(OUT,">>new.dat") or die "Can't open $!";

# which binary to run?
$exe = "./onvort2d";

# how many particles?
$nump = 1000000;
# desired mean velocity error
$desirederror = 1.e-3;
$logdesired = log($desirederror);
# how many runs over which to average the error?
# for 40k and over, 1 is enough
# for 10k, maybe 4?
# for 1k, seems like 16 is barely enough
$runs = 5;

# bucket sizes
@bucket = (128);
# define the order range
#@order = (1,2,3,4,5,6,7,8,9,10);
@order = (2,3,4,5);
#@order = (6,7,8);
# for a given set of the above variables, find theta to minimize CPU time

foreach $thisbucket (@bucket) {
  if ($thisbucket < $nump) {
  foreach $thisorder (@order) {

    # find the rifa that returns the right error!
    $rifa = -1.0;
    $noupperbound = 1;
    $nolowerbound = 1;
    $errordiff = 1.0;
    $numiters = 0;
    $maxiters = 20;	# after 6 we really should have something close to the solution
    print "Trying $thisway $thisbucket $thisorder\n";

    # run until we're within 2% of desired error
    while ($errordiff > 0.02*$desirederror and $numiters < $maxiters) {

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
          # if we don't have an upper bound, double rifa
          $rifa = $rifa * 2.0;
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
          # if we don't have an lower bound, halve rifa
          $rifa = $rifa * 0.5;
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
      print "  upper $urifa   lower $lrifa\n";

      # run the command a few times
      unlink "tempout";
      $command = "${exe} -n=$nump -t=$rifa -o=$thisorder >> tempout";
      #print $command."\n";
      for (my $irun=0; $irun<$runs; $irun=$irun+1) { system $command; }

      # parse tempout

      # look for velocity errors
      @lines = `grep \"error in treecode3\" tempout`;
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
      $thiserror = $meansum;
      # avoid problems taking log of zero
      if ($thiserror > 1.e-8) {
        $logthiserror = log($thiserror);
      } else {
        $logthiserror = log(1.e-8);
      }
      #print $maxsum." ".$thiserror."\n";
      print "  theta ".$rifa." returned mean vel error ".$thiserror."\n";

      # how close is this error to the desired error?
      $errordiff = $desirederror - $thiserror;
      if ($errordiff < 0.0) { $errordiff = -$errordiff; }
      #print "  errordiff ".$errordiff."\n";

      # make sure we don't iterate too much
      $numiters++;
    }

    # we like the error, so let's keep the settings
    # look for total time now
    @lines = `grep \"treecode3 total\" tempout`;
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
    $res = $nump.$sep.$thisbucket.$sep.$thisorder.$sep.$rifa.$sep.$maxsum.$sep.$thiserror.$sep.$treetime.$sep.$dirtime."\n";
    #print $res;
    print "  errors, times: $maxsum / $thiserror  $treetime / $dirtime\n";
    print OUT $res;

    #exit(0);
  } # end foreach @order
  } # end if thisbucket < nump
} # end foreach @bucket

