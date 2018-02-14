#! /usr/bin/env python

import sys
import subprocess
import logging
import optparse
import traceback

if __name__ == '__main__':
    try:
        import container_support as cs
        cs.configure_logging()
        logging.info("running container entrypoint")

        parser = optparse.OptionParser()
        (options, args) = parser.parse_args()

        modes = {
            "train": cs.Trainer.start,
            "serve": cs.Server.start
        }

        if len(args) != 1 or args[0] not in modes:
            raise ValueError("Illegal arguments: %s" % args)

        mode = args[0]
        logging.info("starting %s task", mode)
        modes[mode]()
    except Exception as e:
        trc = traceback.format_exc()
        message = 'uncaught exception: {}\n{}\n'.format(e, trc)
        logging.error(message)
