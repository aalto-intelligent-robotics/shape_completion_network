import numpy as np
import torch as tc

from shape_reconstruction.utils import network_utils, shape_completion_utils, file_utils


def train_model(args):
    network_utils.setup_env(args.use_cuda)
    model = network_utils.setup_network(args)
    model.set_mode(train=True)

    net_dump_folder = "networks/"

    file_utils.create_folder("%scheckpoints/" % (net_dump_folder))

    start_epoch = 0
    if args.net_recover:
        start_epoch = network_utils.recover_network(args, model)
    else:
        network_utils.save_network(model, args,
                                   net_dump_folder + "checkpoints/",
                                   "_epoch_" + str(0))

    dataloader = network_utils.load_dataset(args.num_workers,
                                            args.batch_size,
                                            args.mode,
                                            debug=args.debug)
    print("Starting training (%d samples, %d epochs)" %
          (len(dataloader.dataset), args.num_epochs))

    file_utils.create_folder(net_dump_folder + "training_results/")
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        running_loss = []
        print("EPOCH NUMBER " + str(epoch))
        for idx, data in enumerate(dataloader):
            inputs, targets = network_utils.get_train_data(data, args.use_cuda)
            num_data = inputs.size(0)
            loss = model.train_network(inputs, targets)
            running_loss.append(loss.item() / num_data)
        if (epoch + 1) % args.checkpoint_interval == 0:
            network_utils.save_network(model, args,
                                       net_dump_folder + "checkpoints/",
                                       "_epoch_" + str(int(epoch)))
        file_utils.log_data_to_file(
            np.mean(running_loss), net_dump_folder + "training_results/loss_" +
            args.network_model + ".txt")

    model.eval()

    print("Finished training")

    # Saving final network to file
    print("Saving final network parameters")
    network_utils.save_network(model, args, net_dump_folder + "final_network/")
    tc.cuda.empty_cache()


def test_model(args):
    network_utils.setup_env(args.use_cuda)
    model = network_utils.setup_network(args)
    network_utils.load_parameters(model, args.net_recover_name,
                                  args.network_model)
    model.eval()
    model.apply(network_utils.activate_dropout)
    # When loading a dataset for testing we need to process the data one at a time
    dataloader = network_utils.load_dataset(1, 1, args.mode, debug=args.debug)

    test_cases = shape_completion_utils.get_test_cases(args.test_cases)

    for test_case in test_cases:
        dataloader.dataset.set_model(test_case)
        for idx, data in enumerate(dataloader):
            inputs, targets, observed_pc, object_name, object_pose = network_utils.get_test_data(
                data, args.use_cuda)
            loss, predictions = model.test(inputs,
                                           targets,
                                           num_samples=args.num_test_samples)
            network_utils.save_test_output(predictions, object_name[0],
                                           observed_pc, object_pose, test_case,
                                           args.save_location,
                                           args.network_model,
                                           args.save_voxel_grid,
                                           args.save_samples, args.save_mesh)

            if idx == args.num_objects_to_test:
                break
