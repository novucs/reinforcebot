import React from 'react';
import {Button, Container, Divider, Grid, Header, Image, Input, Segment} from "semantic-ui-react";
import logo from '../icon.svg'
import TopMenu from "../components/TopMenu";
import Footer from "../components/Footer";
import {fetchMe} from "../Util";
import {CopyToClipboard} from "react-copy-to-clipboard/lib/Component";
import {SemanticToastContainer, toast} from "react-semantic-toasts";
import clientLogin from "../client-login.png";
import clientDetail from "../client-detail.png";
import clientList from "../client-list.png";


class CopyableCommand extends React.Component {
  copiedPopup = () => {
    toast(
      {
        type: 'success',
        title: 'Command Copied',
        description: <p>Paste into a terminal to execute</p>
      },
    );
  };

  render = () => (
    <CopyToClipboard text={this.props.command} onCopy={() => this.copiedPopup()}>
      <span>
        <Input
          action={{
            color: 'teal',
            labelPosition: 'right',
            icon: 'copy',
            content: 'Copy',
          }}
          style={{width: '100%'}}
          defaultValue={this.props.command}
        />
      </span>
    </CopyToClipboard>
  )
}


export default class Start extends React.Component {
  constructor(props) {
    super(props);
    this.state = {}
  }

  componentDidMount = () => {
    fetchMe(me => this.setState({me}));
  };

  render = () => (
    <div className='SitePage'>
      <TopMenu me={this.state.me}/>
      <SemanticToastContainer position='bottom-right'/>
      <Container text className='SiteContents' style={{marginTop: '7em', textAlign: 'left'}}>
        <Header as="h2" color="teal" textAlign="center">
          <img src={logo} alt="logo" className="image"/>{" "}
          Getting Started
        </Header>
        <Segment style={{marginBottom: '32px'}}>
          <h3>What is ReinforceBot?</h3>
          <Divider/>
          <p>
            ReinforceBot is a reinforcement learning toolkit that learns how to
            automate any task on any desktop application. It learns by looking
            at how you attempt to solve the problem and asking for feedback on
            its current experience of interacting with the application. Treat
            it as an automated macro recorder that can handle fuzzy scenarios,
            even where things may break.
          </p>
          <p>
            Currently ReinforceBot supports recording, learning from, and
            performing keyboard interactions on any desktop application.
            Xorg on Linux is supported and more desktop environments are aimed
            to be supported in the future. All testing has been limited to
            Manjaro GNOME, so there may be unresolved issues on other
            distributions for this early development stage prototype.
          </p>
          <h3>Installation</h3>
          <Divider/>
          <h4>Quick and easy</h4>
          <p>
            This installation method runs a script with root privileges. It is
            advisable to check scripts before running them. You may download this
            script <a href='/blobs/install.sh'>here</a>.
          </p>
          <CopyableCommand command={'sudo sh -c "$(wget -O- https://reinforcebot.novucs.net/blobs/install.sh)"'}/>
          <h4>Manual</h4>
          <p>
            Download and extract the package below, then run the executable
            found within. Install wherever you like.
          </p>
          <Button
            primary
            download
            href='/blobs/reinforcebot-client.tar.gz'
            icon='download'
            content='Download'
            size='tiny'
          />
          <h3>Usage</h3>
          <Divider/>
          <h4>Starting Up</h4>
          <p>
            As soon as you open the application, you will be greeted with a
            login screen. There is no need to create an account yet, as you can
            optionally continue with offline mode. However, if you wish to make
            use of the online services for backing up trained models or using
            the cloud compute runners then please <a href='/signup'>sign up</a>.
          </p>
          <h4>Agent Selection</h4>
          <p>
            Passing the login screen will take you to the agents listing page,
            which displays all agents available to your account. You can also
            see them <a href='/agents'>online</a>. In this menu you have the
            option to either view an agent details or create a new agent. Agent
            creation is completely free, so feel free to create as many as you
            like to solve whatever problems you are facing!
          </p>
          <h4>Agent Training</h4>
          <p>
            Once you have either selected to create or reuse an existing agent,
            you may now begin training it on experience with your own desktop.
            First you must select an area of your screen to record by either the
            "Select Area" or "Select Window" buttons. If your agent is new and
            has not learned what keys to press, you must record yourself
            interacting with the target application by pressing the "Record"
            button. Be sure to press all the keys that the agent will need to
            use on the first recording session! Then the agent can be let loose
            on your system by pressing the "Handover Control" button, sit back
            and watch it try to figure out what is going on. You may tell the
            agent what it is doing right or wrong at any point by pressing F3.
          </p>
          <h4>Configuration</h4>
          <p>
            The configuration file will be created on first startup of the
            desktop client. It is able to be accessed via your preferred text
            editor, e.g.
          </p>
          <CopyableCommand command={'gedit ~/ReinforceBot/config.json'}/>
          <h3>Screenshots</h3>
          <Divider/>
          <p>
            Desktop client screenshots of login (left), agent detail (center),
            and agent listing (right).
          </p>
          <Grid verticalAlign='middle' columns={3}>
            <Grid.Column>
              <Image size='medium' src={clientLogin}/>
            </Grid.Column>
            <Grid.Column>
              <Image size='medium' src={clientDetail}/>
            </Grid.Column>
            <Grid.Column>
              <Image size='medium' src={clientList}/>
            </Grid.Column>
          </Grid>
          <h3>Uninstall</h3>
          <Divider/>
          <p>
            If you choose to uninstall the ReinforceBot prototype, the process
            is a simple deletion of the program and config files.
          </p>
          <h4>Delete Program Files</h4>
          <CopyableCommand
            command={'sudo rm -rf /opt/reinforcebot /usr/local/share/applications/reinforcebot.desktop'}/>
          <h4>Delete Program Data</h4>
          <CopyableCommand command={'rm -rf ~/ReinforceBot'}/>
        </Segment>
      </Container>
      <Footer/>
    </div>
  );
}
